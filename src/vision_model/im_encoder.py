import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models import resnet50, resnet18, ResNet50_Weights, densenet121, DenseNet121_Weights
from utils.run_configurations import PRETRAIN_CNN, USE_DEFAULT_WEIGHTS, IM_MODDEL, NO_IM_MODEL_CLASSES, TRAIN_FULL_MODEL

class ImageEncoder(nn.Module):
    """ Image feature extractor. Takes images of 512 by 512 actually 224 by 224""" 
    def __init__(self, return_feature_vectors=False, single_disease = False):
        super().__init__()
        # boolean to specify if feature vectors should be returned after roi pooling inside RoIHeads
        self.return_feature_vectors = return_feature_vectors
        self.out_classes = 14 #NO_IM_MODEL_CLASSES
        if single_disease:
            self.out_classes = 1
        if USE_DEFAULT_WEIGHTS:
            print("Using default weights")
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        else:
            print("Using randomly initilaised weights for resnet.")
            resnet = resnet50() #2048 output features
            #resnet = resnet18() #512 output features
            self.mod_feats = 2048 #512
            densenet = densenet121()
            if single_disease:
                print("loading single disease resnet18 model")
                resnet = resnet18()
                self.mod_feats = 512
            
        self.im_model = IM_MODDEL
        if self.im_model == "resnet":
            del densenet
            print('resnet')
            no_features = 512
            #resnet = resnet50()
            # since we have grayscale images, we need to change the first conv layer to accept 1 in_channel (instead of 3)
            resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # use only the feature extractor of the pre-trained classification model
            # (i.e. use all children but the last 2, which are AdaptiveAvgPool2d and Linear)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            #out_features = list(self.backbone)[-1].out_channels
            # for ResNet-50, the number of output channels of the backbone is 2048 (with feature maps of size 16x16) [B,2048,16,16]
            self.feature_extractor = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to reduce spatial dimensions to 1x1 could change to 4x4 
                nn.Flatten(),  # Flatten the tensor [B,2048,1,1] -> [B,2048]
                #nn.Linear(2048, 1024),
                #nn.ReLU(),
                nn.Linear(in_features=self.mod_feats, out_features=no_features)#1024 [B,2048,1,1] -> [B,512,1,1] (no 1,1 if flatten) could have [B,2048,4,4] -> [B,512,4,4]
            )
            self.classifier= nn.Sequential(
                #nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to reduce spatial dimensions to 1x1
                #nn.Flatten(),  # Flatten the tensor
                #nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(in_features=no_features, out_features=self.out_classes), # Fully connected layer with C outputs for classification
                nn.Sigmoid()  # Sigmoid activation for disjoint classification
            )
            total_layers = len(list(resnet.children()))
            print("Total number of layers:", total_layers)
        
        if self.im_model == "densenet":
            del resnet
            print("densenet")
            densenet = densenet121() 
            densenet.features[0] = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone = nn.Sequential(*list(densenet.children())[:-1]) #output 1024 features [64, 1024, 7, 7]
            self.feature_extractor= nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten())
            self.classifier= nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_features=densenet.classifier.in_features, out_features=NO_IM_MODEL_CLASSES), # Fully connected layer with C outputs for classification
                nn.Sigmoid()  # Sigmoid activation for disjoint classification
            )

        #for name, param in self.backbone.named_parameters(): #check trainable
        #    print(f"Parameter: {name}, Requires grad: {param.requires_grad}") #yep all true

    def forward(self, images: Tensor):

        features = self.backbone(images) #[batch size, 2048, 16, 16])
        features = self.feature_extractor(features)
        #print("feats: ", features.shape)
        # if we return region features, then we train/evaluate the full model (with object detector as one part of it)
        if self.return_feature_vectors: 
            #features = self.feature_extractor(features) 
            return features 
        else:
            classified = self.classifier(features)
            return features, classified 
