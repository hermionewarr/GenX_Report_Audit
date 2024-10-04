#downsize images to IMAGE_INPUT_SIZE. to avoid having to to this when loading the data in training
#optional - if use then can comment out the resize in the dataloader.
from PIL import Image
#import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

sys.path.insert(0,  os.path.abspath(".."))
print(sys.path)
from utils.utils import ResizeWithAspectRatio
from utils.path_datasets_and_weights import path_full_dataset 
from utils.run_configurations import *
from datasets import Dataset

IMAGE_INPUT_SIZE =224

transforms = ResizeWithAspectRatio(IMAGE_INPUT_SIZE)
counter = 0

def get_datasets(path_full_dataset):
    usecols = [
        "mimic_image_file_path",
        "labels",
        "image_id"
        #"weight/freq"
    ]
    datasets_as_dfs = {}
    datasets_as_dfs["train"] = pd.read_csv(os.path.join(path_full_dataset, FULL_DATASET_NAME), usecols=usecols) 
    datasets_as_dfs["valid"] = pd.read_csv(os.path.join(path_full_dataset, FULL_VAL_DATASET_NAME), usecols=usecols) 
    datasets_as_dfs["test"] = pd.read_csv(os.path.join(path_full_dataset, FULL_TEST_DATASET_NAME), usecols=usecols)
  
    print(f'val data length {len(datasets_as_dfs["valid"])}, train data length {len(datasets_as_dfs["train"])}')
    raw_train_dataset = pd.concat([datasets_as_dfs["train"], datasets_as_dfs["valid"], datasets_as_dfs["test"]], ignore_index=True)
    print(f"len of df: \n {len(raw_train_dataset)}")
    raw_train_dataset = Dataset.from_pandas(raw_train_dataset)
    
    return raw_train_dataset

def load_n_save_image(dataset_df, index):
    image_path = dataset_df[index]["mimic_image_file_path"]
    try:
        save_path, extension = os.path.splitext(image_path)
        image_path = save_path[:-6] + ".jpg"
        image=Image.open(image_path)
        #print("im size: ", image.size)
        # apply transformations to image
        transformed_image = transforms(image)#["image"] only need dict key if using albumentations
        #print("trans_im: ", transformed_image.size)
        
        #only take the image paths that dont have _small in them
        if "_small" not in save_path:
            save_path =save_path+"_small.jpg"
        else:
            save_path =save_path+".jpg"
        #print(save_path)
        transformed_image.save(save_path)
        #save_path =image_path+"_small.jpg"
        #print(save_path)
        #plt.imshow(transformed_image)
        #transformed_image.save(save_path)
    except Exception as e:
            print(f"__getitem__ failed for: {image_path}")
            print(f"Reason: {e}")
            return None

dataset = get_datasets(path_full_dataset)

for im in range(len(dataset)):
    load_n_save_image(dataset, im)
    """ counter +=1
    print(counter)
    if counter == 2:
        break """
