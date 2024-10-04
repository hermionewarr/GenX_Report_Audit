#import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import ast
import pandas as pd
import os
from utils.run_configurations import PRETRAIN_CNN, PRETRAIN_LLM, TRAIN_FULL_MODEL, NO_IM_MODEL_CLASSES, USE_FINDINGS

class CustomImageDataset(Dataset):
    def __init__(self, dataset_df, log, transforms):
        super().__init__()
        self.dataset_df = dataset_df
        self.transforms = transforms
        self.log =log

    def __len__(self):
        return len(self.dataset_df)
    
    def get_targets(self, index):
        labels = ast.literal_eval(self.dataset_df[index]['labels'])
        return labels
    
    def get_ids(self,index):
        ids = self.dataset_df[index]["image_id"]
        return ids

    def __getitem__(self, index):
        # if something in __get__item fails, then return None
        # collate_fn in dataloader filters out None values
        try:
            image_path = self.dataset_df[index]["mimic_image_file_path"]
            image=Image.open(image_path)
            if NO_IM_MODEL_CLASSES > 1:
                labels = self.get_targets(index)#ast.literal_eval(self.dataset_df[index]['labels'])
            else:
                labels = self.dataset_df[index]['labels']
            #print("lab: ", labels.dtype)
            #print("im size: ", image.size)
            # apply transformations to image
            #transformed = self.transforms(image=image)
            transformed_image = self.transforms(image)#["image"] only need dict key if using albumentations
            #print("trans_im: ", transformed_image.size())
            sample = {
                "image": transformed_image,
                "labels": torch.tensor(labels, dtype=torch.int64),
            }

        except Exception as e:
            self.log.error(f"__getitem__ failed for: {image_path}")
            self.log.error(f"Reason: {e}")
            return None

        return sample
    

class CustomLLM_Dataset(Dataset):
    def __init__(self, dataset_df, log, transforms):
        super().__init__()
        self.dataset_df = dataset_df
        self.transforms = transforms
        self.log =log

    def __len__(self):
        return len(self.dataset_df)
    
    def get_targets(self, index):
        labels = ast.literal_eval(self.dataset_df[index]['labels'])
        return labels

    #add small to each mimic image file path before the extension 
    def remove_suffix_to_path(self, path):
        base_path, extension = os.path.splitext(path)
        # remove _small from base_path
        base_path = base_path[:-6]
        return f"{base_path}{extension}"
    
    def __getitem__(self, index):
        # if something in __get__item fails, then return None
        # collate_fn in dataloader filters out None values
        try:
            if PRETRAIN_CNN or TRAIN_FULL_MODEL:
                image_path = self.dataset_df[index]["mimic_image_file_path"]
                image_path = image_path
                image=Image.open(image_path)
                
                if NO_IM_MODEL_CLASSES > 1:
                    labels = self.get_targets(index)
                else:
                    labels = self.dataset_df[index]['labels']
                # apply transformations to image
                transformed_image = self.transforms(image)
                sample = {
                    "image": transformed_image,
                    "labels": torch.tensor(labels, dtype=torch.int64),
                }

            if TRAIN_FULL_MODEL:

                sample["reference_report"] = self.dataset_df[index]["findings" if USE_FINDINGS else "impression"]
                sample["input_ids"] = self.dataset_df[index]["input_ids"]
                sample["attention_mask"] = self.dataset_df[index]["attention_mask"]
                sample["mimic_image_file_path"] = self.dataset_df[index]["mimic_image_file_path"]
                #print(sample['reference_report'])
                #print(sample['labels'])

        except Exception as e:
            self.log.error(f"__getitem__ failed for: {image_path}")
            self.log.error(f"Reason: {e}")
            return None

        return sample
    
    
    


