"""
Define configurations for training run of full model.
"""

RUN = 40
IM_RUN = 32 #the pretrained image model run that you want to load as the image encoder for the full model
CUDA_DEVICE = 0
RUN_COMMENT = """Clean test"""
CONTINUE_TRAINING = False
#params for image model
PRETRAIN_CNN = False #True 
TEST_CNN = False
OVERSAMPLING = False #true for training image model, false currently for full model
NO_IM_MODEL_CLASSES = 14 # 1 when training for single disease for the modular ACs
USE_DEFAULT_WEIGHTS = False #whether to initilaise with resnet trained on imagenet weights
IM_MODDEL = 'resnet' #'densenet'
SD_DISEASE = 'Consolidation'
SD_RUN = 43
SEED = 1 # if you want to set it
#imge model data
#uncertain chexpert labels considered as positive, not mentioned negative
if NO_IM_MODEL_CLASSES == 1:
    
    IM_DATASET_NAME = f"train_disease_labels_oversampled.csv"  #have the freq col - oversampled should be true
    IM_VAL_DATASET_NAME = f"val_disease_labels_oversampled.csv" 
    IM_TEST_DATASET_NAME= f"test_disease_labels_oversampled.csv" 

else:
    IM_DATASET_NAME = "example_data/im_data_example.csv" 
    IM_VAL_DATASET_NAME = "example_data/im_data_example.csv" 
    IM_TEST_DATASET_NAME= "example_data/im_data_example.csv"

IMAGE_INPUT_SIZE = 224 #512
#params for language model
PRETRAIN_LLM = False
#params for full model
TRAIN_FULL_MODEL = True
UPDATE_IMAGE_MODEL = True #whether to update the image model while training the full model. If False, the image model will be frozen.
FULL_DATASET_NAME = "example_data/full_model_data_example.csv" 
FULL_VAL_DATASET_NAME = "example_data/full_model_data_example.csv" 
FULL_TEST_DATASET_NAME = "example_data/full_model_data_example.csv" 
NO_IMAGE_TOKENS = 10
USE_FINDINGS = False
#params for training
PERCENTAGE_OF_TRAIN_SET_TO_USE = 1
PERCENTAGE_OF_VAL_SET_TO_USE = 1
PERCENTAGE_OF_TEST_SET_TO_USE = 1
#BATCH_SIZE = 512 #for image model #32 #512 #for vlp: #48 #32 
BATCH_SIZE = 48 #for full model
#EFFECTIVE_BATCH_SIZE = #32 #64   # batch size after gradient accumulation
NUM_WORKERS = 8 #10
NUM_EPOCHS = 100 #200
FULL_MODEL_EPOCHS = 40
LR = 6e-4 #6e-4 #1e-4 for BS 8
IM_LR = 1e-3
EARLY_STOPPING  = 50
# how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
EVALUATE_EVERY_K_BATCHES = 4000 #4000 <- for full model #400 <- for im model (bs 512) 

#TEXT GENERATION PARAMS
# MAX_NUM_TOKENS_GENERATE is set arbitrarily to 300. Most generated sentences have around 70 tokens,
# so this is just an arbitrary threshold that will never be reached if the language model is not completely untrained (i.e. produces gibberish)
MAX_NUM_TOKENS_GENERATE = 300
