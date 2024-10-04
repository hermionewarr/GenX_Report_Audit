import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import logging
from utils.utils import metrics, disease_confusion_matrix, get_torch_transforms, get_im_model
from utils.run_configurations import *
from utils.custom_dataset import CustomImageDataset
from utils.path_datasets_and_weights import path_full_dataset_im
from datasets import Dataset

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
#PERCENTAGE_OF_TEST_SET_TO_USE = 0.1

def test_model(model, test_dl, save_path, log):
    IM_LOSS = nn.BCELoss()
    model.eval()
    test_output = []
    test_prob_output = []
    test_targets = []
    with torch.no_grad():
        for no, batch in tqdm(enumerate(test_dl)):
            images, targets = batch.values()
            feat, output = model(images.to(device))
            if NO_IM_MODEL_CLASSES == 1:
                targets = targets.unsqueeze(-1)

            test_loss = IM_LOSS(output, targets.float().to(device))
     
            prob_output = output.detach().cpu()
            output = (prob_output>0.5).float()
            #output = (output.detach().cpu()>0.5).float()
            test_output.append(output)
            test_prob_output.append(prob_output)
            test_targets.append(targets.detach().cpu())

    test_metrics, test_targets, test_output, test_prob_output, roc_curves = metrics(test_targets, test_output, return_data=True, prob_outputs=test_prob_output)
    test_metrics.to_csv(os.path.join(save_path, "metrics.csv"))
    log.info(test_metrics)
    av = test_metrics.mean(axis=1)
    log.info(av)
    conf_path = os.path.join(save_path, "conf_matrix.png")
    f = disease_confusion_matrix(test_targets, test_output, conf_path)
    return test_metrics, f, roc_curves

def get_one_dataset(path_full_dataset, data_path, log):
    usecols = [
        "mimic_image_file_path",
        "labels",
    ]
    dataset_as_dfs = pd.read_csv(os.path.join(path_full_dataset, data_path), usecols=usecols) 
    total_num_samples_train = len(dataset_as_dfs)
    # compute new number of samples for both train and val
    new_num_samples_train = int(PERCENTAGE_OF_TEST_SET_TO_USE * total_num_samples_train)

    log.info(f"test: {new_num_samples_train} images")
    
    # limit the datasets to those new numbers
    dataset_as_dfs = dataset_as_dfs.sample(n=new_num_samples_train, random_state=SEED)
    raw_train_dataset = Dataset.from_pandas(dataset_as_dfs)

    test_dataset = CustomImageDataset(raw_train_dataset, log, transforms=get_torch_transforms("val"))
    
    return test_dataset


def test_main(device, log, test=False):
    log.info(f"Run: {RUN}")
    test_dataset = get_one_dataset(path_full_dataset_im, IM_TEST_DATASET_NAME, log)

    checkpoint_path = f"/home/hermione/Documents/VLP/GenX/src/runs/image_model/run_{RUN}/checkpoints/"
    #checkpoint = os.path.join(checkpoint_path,os.listdir(checkpoint_path)[0])
    if CONTINUE_TRAINING == True:
        checkpoint_path = f"/home/hermione/Documents/VLP/GenX/src/runs/image_model/run_{RUN}/cont/checkpoints/"
    f1_checkpoint = [file for file in os.listdir(checkpoint_path) if "f1" in file][0]
    checkpoint = os.path.join(checkpoint_path,f1_checkpoint)
    
    model = get_im_model(device, checkpoint)
    test_dl = DataLoader(
        test_dataset,
        batch_size= BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        persistent_workers=False,
    )
    
    save_path = f"/home/hermione/Documents/VLP/GenX/src/runs/image_model/run_{RUN}/"
    if CONTINUE_TRAINING == True:
        save_path = f"/home/hermione/Documents/VLP/GenX/src/runs/image_model/run_{RUN}/cont/"
    if test:
        save_path = f"/home/hermione/Documents/VLP/GenX/src/runs/image_model/run_{RUN}/test/"
    
    log.info(f"save paths: {save_path}")
    
    test_metrics, conf_mat, roc_curves = test_model(model, test_dl, save_path, log)
    roc_curves.write_image(os.path.join(save_path, "roc_curves.png"))
    return test_metrics, conf_mat, roc_curves

