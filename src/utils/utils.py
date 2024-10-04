import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
import torchvision.transforms.functional as F
from collections import defaultdict
import ast
import re
from transformers import GPT2Tokenizer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp

from vision_model.im_encoder import ImageEncoder
from utils.run_configurations import *
from utils.custom_dataset import *
from utils.path_datasets_and_weights import *
from utils.custom_collator import CustomCollator

from datasets import Dataset #has to be here because another dataset import in utils.custom_dataset

#chexpert labels 
def classes_setup():
    global diseases
    if NO_IM_MODEL_CLASSES ==14:
        diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 
            'Pneumothorax','Support Devices']
    elif NO_IM_MODEL_CLASSES == 1:
        diseases = [SD_DISEASE]
    print(f"diseases: {diseases}")
    return diseases

classes_setup()
#dataset_name =  DATASET_NAME

def train_setup(model, epochs, checkpoints, device, log, lowest_val_loss=100):
    log.info(f"Number of output classes: {NO_IM_MODEL_CLASSES}")
    
    #get train setup # hyperparameters etc
    LLM_LOSS = nn.CrossEntropyLoss()
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    #LLM_OPTIMISER = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    LLM_OPTIMISER = torch.optim.AdamW(model.parameters(), lr=LR, betas=(beta1, beta2), weight_decay =weight_decay)
    if PRETRAIN_CNN:
        if NO_IM_MODEL_CLASSES == 1:
            weights=torch.tensor([1, 0.024], dtype=torch.float)
            IM_LOSS = nn.BCELoss() 
        else:
            weights = cal_loss_weights(log)
            weights=torch.tensor([weights], dtype=torch.float)
            IM_LOSS = nn.BCELoss(weight=weights.to(device)) 
    else:
        IM_LOSS = nn.BCELoss() #not used make this better.
    IM_OPTIMISER = optim.Adam(model.parameters(), lr=IM_LR) #make this AdamW. betas=(0.9, 0.98), eps=1e-9 <- TUM run<17, betas=(0.9, 0.999), eps=1e-08 <-default runs>16.
    
    IM_INDIV_LOSS = nn.BCELoss(reduction='none') 

    run_params = {}
    run_params["epochs"] = epochs
    run_params["checkpoints_folder_path"] = checkpoints
    run_params["lowest_val_loss"] = lowest_val_loss
    run_params["best_epoch"] = None  # the epoch with the lowest val loss overall
    run_params["overall_steps_taken"] = 0 #overall_steps_taken  # for logging to tensorboard
    run_params["LLM_OPTIMISER"] = LLM_OPTIMISER
    run_params["LLM_LOSS"] = LLM_LOSS
    run_params["IM_OPTIMISER"] = IM_OPTIMISER
    run_params["IM_LOSS"] = IM_LOSS
    run_params["IM_INDIV_LOSS"] = IM_INDIV_LOSS
    run_params["NO_IM_MODEL_CLASSES"] = NO_IM_MODEL_CLASSES
    return run_params

def get_im_model(device, checkpoint=None):
    
    if PRETRAIN_CNN or TEST_CNN:
        print("Loading image classifier model.")
        model = ImageEncoder()
        if NO_IM_MODEL_CLASSES == 1:
            print("loading sd img model")
            model = ImageEncoder(return_feature_vectors=False, single_disease=True)
    elif TRAIN_FULL_MODEL:
        print("Loading image feature model.")
        model = ImageEncoder(return_feature_vectors=True)

    model.to(device, non_blocking=False)

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint)["model"])
    model.train()

    return model

def get_checkpoint(RUN):
    """ return the checkpoint path"""
    im_checkpoint_path = f'/home/hermione/Documents/VLP/GenX/src/runs/image_model/run_{RUN}/checkpoints/'
    f1_checkpoint = [file for file in os.listdir(im_checkpoint_path) if "f1" in file][0]
    im_checkpoint = os.path.join(im_checkpoint_path,f1_checkpoint)
    last_saved_epoch = int(re.search(r'epoch_(\d+)', f1_checkpoint).group(1))
    return im_checkpoint, last_saved_epoch

def get_full_checkpoint(run):
    """ return the checkpoint path"""
    checkpoint_path = f'/home/hermione/Documents/VLP/GenX/src/runs/full_model/run_{run}/checkpoints/'
    checkpoint = [file for file in os.listdir(checkpoint_path) if "val" in file][0]
    checkpoint = os.path.join(checkpoint_path,checkpoint)
    #last_saved_epoch = int(re.search(r'epoch_(\d+)', f1_checkpoint).group(1))
    return checkpoint #, last_saved_epoch

def get_dataloaders(train_dataset, val_dataset, log, tokenizer = None):
    if PRETRAIN_CNN:
        if OVERSAMPLING ==False:
            log.info("no oversampling")
            train_loader = DataLoader(
                train_dataset,
                batch_size= BATCH_SIZE,
                shuffle=True, #true without sample weights_basic
                num_workers=NUM_WORKERS,
                pin_memory=False,
                persistent_workers=True,
            )
        else: #getting sample weighting
            log.info("oversampling")
            weighted_targets = pd.read_csv(f'{path_full_dataset_im}/{IM_DATASET_NAME}')
            train_weights = weighted_targets['Frequency'].to_list()
            weighted_sampler =  WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_dataset),
            replacement=True
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size= BATCH_SIZE,
                shuffle=False, #false with sampler
                num_workers=NUM_WORKERS,
                sampler = weighted_sampler,
                pin_memory=False,
                persistent_workers=True,
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size= BATCH_SIZE,
            shuffle=False, #make false later
            num_workers=NUM_WORKERS, 
            pin_memory=False,
            persistent_workers=False,
        )
    elif TRAIN_FULL_MODEL:
        tokenizer = get_tokenizer()
        train_loader = DataLoader(
                train_dataset,
                collate_fn=CustomCollator(tokenizer),
                batch_size= BATCH_SIZE,
                shuffle =True, #True, #true without sample weights_basic
                num_workers=NUM_WORKERS,
                pin_memory=False,
                persistent_workers=True,
            )
        val_loader = DataLoader(
            val_dataset,
            collate_fn=CustomCollator(tokenizer),
            batch_size= BATCH_SIZE,
            shuffle=False, #make false later
            num_workers=NUM_WORKERS, 
            pin_memory=False,
            persistent_workers=False,
        )

    return train_loader, val_loader

def findings_or_impression():
    return "findings" if USE_FINDINGS else "impression"

def get_datasets(path_full_dataset, config_file_path, log, tokenizer = None):
    usecols = [
        "mimic_image_file_path",
        "labels",
        "image_id"
        #"weight/freq" #for training the image model if oversampled
    ]
    if TRAIN_FULL_MODEL:
        usecols.append(findings_or_impression())
        DATASET_NAME = FULL_DATASET_NAME
        VAL_DATASET_NAME = FULL_VAL_DATASET_NAME
    elif PRETRAIN_CNN:
        DATASET_NAME = IM_DATASET_NAME
        VAL_DATASET_NAME = IM_VAL_DATASET_NAME

    datasets_as_dfs = {}
    datasets_as_dfs["train"] = pd.read_csv(os.path.join(path_full_dataset, DATASET_NAME), usecols=usecols) 
    datasets_as_dfs["valid"] = pd.read_csv(os.path.join(path_full_dataset, VAL_DATASET_NAME), usecols=usecols) #val_white_square_classify.csv

    #train_weights = cal_loss_weights(datasets_as_dfs["train"], log)
    if TRAIN_FULL_MODEL:
        log.info(len(datasets_as_dfs["train"]))
        datasets_as_dfs["train"] = datasets_as_dfs["train"][datasets_as_dfs["train"][findings_or_impression()].notna() & datasets_as_dfs["train"]["labels"].notna()]
        datasets_as_dfs["valid"] = datasets_as_dfs["valid"][datasets_as_dfs["valid"][findings_or_impression()].notna() & datasets_as_dfs["valid"]["labels"].notna()]
        log.info(len(datasets_as_dfs["train"]))

    total_num_samples_train = len(datasets_as_dfs["train"])
    total_num_samples_val = len(datasets_as_dfs["valid"])
    # compute new number of samples for both train and val
    new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)

    log.info(f"Train: {new_num_samples_train} images")
    log.info(f"Val: {new_num_samples_val} images")
    with open(config_file_path, "a") as f:
        f.write(f"\tTRAIN NUM IMAGES: {new_num_samples_train}\n")
        f.write(f"\tVAL NUM IMAGES: {new_num_samples_val}\n")

    # limit the datasets to those new numbers
    if PERCENTAGE_OF_TRAIN_SET_TO_USE <1:
        datasets_as_dfs["train"] = datasets_as_dfs["train"].sample(n=new_num_samples_train, random_state=SEED)
    if PERCENTAGE_OF_VAL_SET_TO_USE <1:
        datasets_as_dfs["valid"] = datasets_as_dfs["valid"].sample(n=new_num_samples_val, random_state=SEED)

    log.info(f'val data length {len(datasets_as_dfs["valid"])}, train data length {len(datasets_as_dfs["train"])}')
    
    print(datasets_as_dfs["train"].columns())
    raw_train_dataset = Dataset.from_pandas(datasets_as_dfs["train"])
    raw_val_dataset = Dataset.from_pandas(datasets_as_dfs["valid"])

    if PRETRAIN_CNN:
        train_dataset = CustomImageDataset(raw_train_dataset, log, transforms=get_torch_transforms("train"))
        val_dataset = CustomImageDataset(raw_val_dataset, log, transforms=get_torch_transforms("val"))
    
    if TRAIN_FULL_MODEL:
        # tokenize the raw datasets
        tokenized_train_dataset, tokenized_val_dataset = get_tokenized_datasets(raw_train_dataset, raw_val_dataset)
        train_dataset = CustomLLM_Dataset(tokenized_train_dataset, log, transforms=get_torch_transforms("train"))#make these work for llm+vis
        val_dataset = CustomLLM_Dataset(tokenized_val_dataset, log, transforms=get_torch_transforms("val"))

    return train_dataset, val_dataset

def get_tokenized_datasets(raw_train_dataset, raw_val_dataset):
    tokenizer = get_tokenizer()
    def tokenize_function(example):
        phrases = example[findings_or_impression()]
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        phrases_with_special_tokens = [bos_token + phrases + eos_token]

        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return tokenizer(phrases_with_special_tokens, truncation=True, max_length=256)#1024

    tokenized_train_dataset = raw_train_dataset.map(tokenize_function)
    tokenized_val_dataset = raw_val_dataset.map(tokenize_function)
    # tokenized datasets will consist of the columns
    #   - mimic_image_file_path (str)
    #   - input_ids (List[List[int]])
    #   - attention_mask (List[List[int]])
    #   - findings (str)
    #   - labels (List[int])

    return tokenized_train_dataset, tokenized_val_dataset

def get_tokenizer():
    checkpoint = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_torch_transforms(dataset: str):
    # see compute_mean_std_dataset.py in src/dataset do it ur self b
    mean = [0.471]
    std = [0.302]
    #mean=[0.485, 0.456, 0.406]
    #std=[0.229, 0.224, 0.225]
    train_transforms = v2.Compose([
        ResizeWithAspectRatio(IMAGE_INPUT_SIZE), #make the image square amd pad to keep proportions - can remove if downsized and saved the CXRs to save time when training
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True), # [0, 1].
        ###
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 5.))], p=0.25),
        v2.RandomApply([v2.RandomResizedCrop(size=IMAGE_INPUT_SIZE, scale=(0.8,1), ratio=(0.8,1.2), antialias=True)],p=0.5),
        v2.ColorJitter(brightness = 0.3, contrast = 0.3),
        v2.RandomRotation(90),
        v2.RandomHorizontalFlip(p=0.25),
        v2.RandomVerticalFlip(p=0.25),
        ###
        #v2.ToTensor(),
        #v2.Normalize(mean, std), #mean=mean, std=std)
        lambda x: (x-0.5)*6,
        #lambda x: x.expand(3,-1,-1),
        ]
    )

    # don't apply data augmentations to val set (and test set)
    val_transforms = v2.Compose([
            ResizeWithAspectRatio(IMAGE_INPUT_SIZE),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            #v2.Resize(size= IMAGE_INPUT_SIZE, antialias=True),
            #v2.CenterCrop(IMAGE_INPUT_SIZE), 
            #v2.ToTensor(),
            #v2.Normalize(mean=mean, std=std),#mean=mean, std=std #check this!
            lambda x: (x-0.5)*6,
            #lambda x: x.expand(3,-1,-1),
        ]
    )

    if dataset == "train":
        return train_transforms
    else:
        return val_transforms

#fluff around model   
def create_run_folder(log, path_runs):
    """
    Run folder will contain:
        - a folder called "checkpoints" for the saved checkpoints
        - a folder called "generated_sentences_and_reports" that store the generated sentences and reports
        which were created at each evaluation
        - a txt file called "run_config", which stores the information specified in run_configurations.py
    """
    run_folder_path = os.path.join(path_runs, f"run_{RUN}")
    
    if os.path.exists(run_folder_path) and CONTINUE_TRAINING==False:
        log.error(f"Folder to save run {RUN} already exists at {run_folder_path}.")
        log.error("Delete the folder or change the run number.")
        return None
    
    elif os.path.exists(run_folder_path) and CONTINUE_TRAINING==True:
        log.info("Continuing training.")
        if TRAIN_FULL_MODEL == False:
            last_saved_epoch = int(re.search(r'epoch_(\d+)', [file for file in os.listdir(os.path.join(run_folder_path,"checkpoints"))][0]).group(1))
            log.info(f"last saved epoch for f1 checkpoint is: {last_saved_epoch}")
        run_folder_path = (os.path.join(run_folder_path, "cont"))
    
    checkpoints_folder_path = os.path.join(run_folder_path, "checkpoints")
    generated_sentences_and_reports_folder_path = os.path.join(run_folder_path, "generated_sentences_and_reports")
    generated_reports_folder_path = os.path.join(generated_sentences_and_reports_folder_path, "generated_reports")
    log_file = os.path.join(run_folder_path, "log_file")

    os.mkdir(run_folder_path)
    os.mkdir(checkpoints_folder_path)
    os.mkdir(generated_sentences_and_reports_folder_path)
    os.mkdir(generated_reports_folder_path)
 

    log.info(f"Run {RUN} folder created at {run_folder_path}.")

    config_file_path = os.path.join(run_folder_path, "run_config.py")
    if CONTINUE_TRAINING:
        old_config = os.path.join(run_folder_path, "run_config.py") #need to load old as run the acutal run configs but currently you gotta manualy check it sorrry
        config_file_path = os.path.join(run_folder_path, "run_config_cont.py")
        
    config_parameters = {
        "COMMENT": f"{RUN_COMMENT}",
        "CONTINUE_TRAINING": CONTINUE_TRAINING,
        "PRETRAIN_CNN": PRETRAIN_CNN,
        "PRETRAIN_LLM": PRETRAIN_LLM,
        "USE_DEFAULT_WEIGHTS": USE_DEFAULT_WEIGHTS,
        "TRAIN_FULL_MODEL": TRAIN_FULL_MODEL,
        "OVERSAMPLING": OVERSAMPLING,
        "NO_IM_MODEL_CLASSES": NO_IM_MODEL_CLASSES,
        "IM_MODDEL" :f"{IM_MODDEL}",
        "IM_DATASET_NAME": f"{IM_DATASET_NAME}", 
        "IM_VAL_DATASET_NAME": f"{IM_VAL_DATASET_NAME}",
        "IM_TEST_DATASET_NAME": f"{IM_TEST_DATASET_NAME}",
        "IMAGE_INPUT_SIZE": IMAGE_INPUT_SIZE,
        "FULL_DATASET_NAME" : f"{FULL_DATASET_NAME}",
        "FULL_VAL_DATASET_NAME" : f"{FULL_VAL_DATASET_NAME}",
        "FULL_TEST_DATASET_NAME" : f"{FULL_TEST_DATASET_NAME}",
        "PERCENTAGE_OF_TRAIN_SET_TO_USE": PERCENTAGE_OF_TRAIN_SET_TO_USE,
        "PERCENTAGE_OF_VAL_SET_TO_USE": PERCENTAGE_OF_VAL_SET_TO_USE,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "NUM_EPOCHS": NUM_EPOCHS,
        "LR": LR,
        "EVALUATE_EVERY_K_BATCHES": EVALUATE_EVERY_K_BATCHES,
        "MAX_NUM_TOKENS_GENERATE": MAX_NUM_TOKENS_GENERATE
    }

    with open(config_file_path, "w") as f:
        f.write(f"RUN = {RUN}\n")
        for param_name, param_value in config_parameters.items():
            f.write(f"{param_name} = {param_value}\n")

    return checkpoints_folder_path, config_file_path, generated_sentences_and_reports_folder_path, log_file, run_folder_path

def early_stopping(log, f1, highest_f1, bal_acc, highest_bal_acc, run_params, epoch, val_loss, train_loss, optimizer,model, early_stopping_count, start_epoch):
    epoch = epoch+start_epoch
    if bal_acc > highest_bal_acc:
        save_model(run_params, epoch, val_loss, optimizer,model, log, bal_acc=bal_acc)
        log.info(f'bal acc increased to {bal_acc:.3f} from {highest_bal_acc:.3f}. Model Saved.')
        highest_bal_acc = bal_acc
        early_stopping_count = 0

    if f1 > highest_f1:
        save_model(run_params, epoch, val_loss, optimizer,model, log, f1=f1)
        log.info(f'f1 increased to {f1:.3f} from {highest_f1:.3f}. Model Saved.')
        highest_f1 = f1
        early_stopping_count = 0

    elif val_loss < run_params["lowest_val_loss"]:
        save_model(run_params, epoch, val_loss, optimizer,model,log)
        log.info('Val loss decreased. Model Saved.')
        early_stopping_count = 0
    else:
        early_stopping_count +=1
        log.info(f'early stopping count increased to {early_stopping_count}')
        if early_stopping_count == EARLY_STOPPING:
            log.info(f"Early stopping implemented at epoch {epoch+1}.")
            log.info(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Val loss: {val_loss}")
            #maybe save here anyway just in case
            
            
    return highest_f1, highest_bal_acc, early_stopping_count

def save_model(run_params, epoch, val_loss, optimizer, model, log, f1=None, final=False, bal_acc=None):
    run_params["best_epoch"] = epoch
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "current_epoch": epoch,
        "lowest_val_loss": val_loss,
    }

    if run_params["lowest_val_loss"] > val_loss:
        log.info(f"{run_params['lowest_val_loss']}")
        run_params["lowest_val_loss"] = val_loss
        save_path = os.path.join(run_params["checkpoints_folder_path"], f"checkpoint_val_loss_{val_loss:.3f}_epoch_{epoch}.pt")
        for f in os.listdir(run_params["checkpoints_folder_path"]):
            if f.startswith('checkpoint_val'):
                os.remove(os.path.join(run_params["checkpoints_folder_path"],f))
        torch.save(checkpoint, save_path) 
    
    if f1:
        save_path = os.path.join(run_params["checkpoints_folder_path"], f"checkpoint_f1_{f1:.3f}_epoch_{epoch}.pt")
        for f in os.listdir(run_params["checkpoints_folder_path"]):
            if f.startswith('checkpoint_f1'):
                os.remove(os.path.join(run_params["checkpoints_folder_path"],f))
        torch.save(checkpoint, save_path)

    if bal_acc:
        save_path = os.path.join(run_params["checkpoints_folder_path"], f"checkpoint_bal_acc_{bal_acc:.3f}_epoch_{epoch}.pt")
        for f in os.listdir(run_params["checkpoints_folder_path"]):
            if f.startswith('checkpoint_bal_acc'):
                os.remove(os.path.join(run_params["checkpoints_folder_path"],f))
        torch.save(checkpoint, save_path)

    if final == True:
        save_path = os.path.join(run_params["checkpoints_folder_path"], f"checkpoint_final_epoch_{epoch}.pt")
        for f in os.listdir(run_params["checkpoints_folder_path"]):
            if f.startswith('checkpoint_final'):
                os.remove(os.path.join(run_params["checkpoints_folder_path"],f))
        torch.save(checkpoint, save_path)
    return run_params


#val and test loop utils
# def indivi_losses(output, targets, run_params):
#     """diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
#        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
#        'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
#        'Support Devices'] #just to make sure No Findings isnt included here
#     """
#     indiv_losses = run_params["IM_INDIV_LOSS"](output, targets) #[batch_size, 13]
#     indiv_losses = torch.mean(indiv_losses.detach().to('cpu'), dim=1)
    
#     disease_losses = {f'disease_losses/{disease_name} loss': loss.item() for disease_name, loss in zip(diseases, indiv_losses)}

#     return disease_losses

def indiv_losses2(losses, log):
    indiv_losses = torch.stack(losses)
    indiv_losses = torch.mean(indiv_losses, dim=0)
    disease_losses = {f'disease_losses/{disease_name} loss': loss for disease_name, loss in zip(diseases, indiv_losses)}
    return disease_losses

def extend_output(list_data):
    """all zeros except support devices = no findings so add a no findings column if only 13 classes"""
    tensor = torch.cat(list_data, dim=0)
    all_zeros = torch.all(tensor[:,:-1] == 0, dim=1)
    new_column = torch.where(all_zeros.view(-1, 1), torch.ones_like(all_zeros).view(-1, 1), torch.zeros_like(all_zeros).view(-1, 1))
    # Concatenate the new column to the original output tensor
    if NO_IM_MODEL_CLASSES ==13:
        extended_output = torch.cat([tensor, new_column], dim=1)
    else:
        extended_output = tensor
    return extended_output

def metrics(targets, model_output, return_data = False, prob_outputs=None):
    """returns a df of relevant metrics for each disease class """
       
    accuracy = []
    precision = []
    recall = []
    f1 = []
    bal_accuracy = []
    auc = []
    
    val_targets = extend_output(targets)
    val_outputs = extend_output(model_output)
    
    df_outputs = pd.DataFrame(val_outputs, columns=diseases)
    df_targets = pd.DataFrame(val_targets, columns=diseases)
    #log.info(np.unique(df_outputs.values))
    diseases_local = diseases
    if NO_IM_MODEL_CLASSES ==13:
        diseases_local = diseases_local.append('No Finding')

    if prob_outputs is not None:
        prob_outputs = extend_output(prob_outputs)
        df_prob = pd.DataFrame(prob_outputs, columns=diseases_local)
        roc_curve, opt_thresholds = plot_roc_curves(df_targets, df_prob)
        #print(f"fml opt thresholds: {opt_thresholds}")
        df_opt_outputs = df_prob
        for col, threshold in zip(df_prob.columns, opt_thresholds):
            df_opt_outputs[col] = (df_prob[col] > threshold).astype(int)
        #df_outputs =df_opt_outputs

    for disease in diseases_local:
        acc = accuracy_score(df_targets[disease], df_outputs[disease])
        accuracy.append(acc)
        precision.append(precision_score(df_targets[disease], df_outputs[disease], zero_division=0))
        recall.append(recall_score(df_targets[disease], df_outputs[disease], zero_division=0))
        f1.append(f1_score(df_targets[disease], df_outputs[disease], zero_division=0))
        bal_accuracy.append(balanced_accuracy_score(df_targets[disease], df_outputs[disease]))
        if prob_outputs is not None:
            auc.append(roc_auc_score(df_targets[disease], df_prob[disease]))

    metrics = np.vstack((accuracy, precision, recall, f1, bal_accuracy))
    df_metrics = pd.DataFrame(metrics, columns=diseases_local, index = ['accuracy','precision','recall','f1','bal_accuracy'])
    
    if prob_outputs is not None:
        df_metrics = pd.concat([df_metrics,pd.Series(auc, index=df_metrics.columns, name='auc').to_frame().T]) #np.array(auc)

    if return_data and (prob_outputs is not None):
        return df_metrics, df_targets, df_outputs, df_prob, roc_curve
    elif return_data:
        return df_metrics, df_targets, df_outputs
    else:
        return df_metrics

def disease_confusion_matrix(targets, outputs, save_path=None):
    """plot the confusion matrices of each of the disease classes"""
    #conf_matrix = multilabel_confusion_matrix(targets, outputs)
    diseases_local = diseases
    if NO_IM_MODEL_CLASSES ==1:
        disp = ConfusionMatrixDisplay(
            confusion_matrix(targets[diseases_local[0]],outputs[diseases_local[0]]),
            display_labels=[0, 1])
        disp.plot(values_format='.4g')
        disp.ax_.set_title(f'{diseases_local[0]}')
        f = plt.gcf()
    else:
        # Plot the confusion matrix
        f, axes = plt.subplots(2, 7, figsize=(25, 10))
        axes = axes.ravel()
        for i, disease in enumerate(diseases_local):
            
            disp = ConfusionMatrixDisplay(
                confusion_matrix(targets[disease],outputs[disease]),
                display_labels=[0, 1])
            disp.plot(ax=axes[i], values_format='.4g')
            disp.ax_.set_title(f'{disease}')
            if i<7:
                disp.ax_.set_xlabel('')
            if i%7!=0:
                disp.ax_.set_ylabel('')
            disp.im_.colorbar.remove()

        plt.subplots_adjust(wspace=0.10, hspace=0.1)
        f.colorbar(disp.im_, ax=axes)

    #plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    return f

#transforms
#fuck albumentations
class ResizeWithAspectRatio(object):
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # Calculate new size while maintaining the original aspect ratio
        width, height = img.size
        aspect_ratio = width / height
        
        if (width == IMAGE_INPUT_SIZE) and (height == IMAGE_INPUT_SIZE):
            return img
        
        elif width > height:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)

        # Resize the image
        resized_img = img.resize((new_width, new_height))
        #print(resized_img.size)
        # Create a new blank image of the target size
        final_img = Image.new("L", (self.target_size, self.target_size), 0)

        # Paste the resized image onto the center of the blank image
        x_offset = (self.target_size - new_width) // 2
        y_offset = (self.target_size - new_height) // 2
        final_img.paste(resized_img, (x_offset, y_offset))

        return final_img
    
#calculate weights for loss
def cal_loss_weights(log):
    #dont use this when only one image class
    if OVERSAMPLING == False:
        log.info("Calculating weights for loss using formula: N_tot/(C*N_c)")
        
        #load in data
        usecols = [
            "mimic_image_file_path",
            "labels",
            "image_id"
        ]
        train_data = pd.read_csv(os.path.join(path_full_dataset_im, IM_DATASET_NAME), usecols=usecols) 
        #add no findings label
        if NO_IM_MODEL_CLASSES ==14:
            train_data['labels'] = train_data['labels'].apply(lambda x: ast.literal_eval(x))
           
        #Create a defaultdict to store counts for each position
        position_counts = defaultdict(lambda: defaultdict(int))
        
        # Iterate through the lists and count occurrences of 1s and 0s at each position
        for lst in train_data['labels']:
            for idx, val in enumerate(lst):
                position_counts[idx][val] += 1

        # Create a list to store the formatted counts
        formatted_counts = []
        disease_cases = []

        # Generate the formatted counts list
        for idx, counts in position_counts.items():
            formatted_counts.append(f"{counts[1]} 1s at position {idx + 1}, {counts[0]} 0s at position {idx + 1}")
            disease_cases.append([counts[1], counts[0]])

        # calc weights
        df_disease_counts = pd.DataFrame(disease_cases, index=diseases, columns=['Cases','Not cases'])
        df_disease_counts['sum']=df_disease_counts['Cases']+df_disease_counts['Not cases']
        df_disease_counts['class weights'] = (df_disease_counts['Cases'].sum())/(14*df_disease_counts['Cases'])
        
        log.info(df_disease_counts)
        class_weights = df_disease_counts['class weights'].tolist()
        log.info(len(df_disease_counts))

    if OVERSAMPLING:
        log.info("Oversampling. Recalculating loss weights")
        disease_counts = pd.read_csv('/Data/hermione_data/dataset-with-reference-reports/all_im_data/disease_counts.csv')
        disease_counts = disease_counts[diseases]
        disease_counts_processed = disease_counts.transpose().reset_index()   
        disease_counts_processed.columns=['disease', 'Frequency']  
        disease_counts_processed['class weights'] = disease_counts_processed['Frequency'].sum()/(len(disease_counts_processed)*disease_counts_processed['Frequency'])       
        class_weights = disease_counts_processed['class weights'].tolist() 
        print(class_weights)
        
    
    return class_weights

def plot_roc_curves(targets, predicted_probs):
    """
    Plot ROC curves for each class based on target labels and predicted probabilities.

    Parameters:
    - targets: DataFrame, true labels for each class
    - predicted_probs: DataFrame, predicted probabilities for each class

    Returns:
    - fig: Matplotlib figure object
    """
    num_classes = len(targets.columns)
    num_rows = 2
    num_cols = 7 #num_classes // num_rows + num_classes % num_rows 
    #fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 8))
    fig = sp.make_subplots(rows=num_rows, cols=num_cols, subplot_titles=targets.columns)
    #fig = go.Figure()
    opt_thresholds =[]
    for i, class_label in enumerate(targets.columns):
        row_idx = i // num_cols +1
        col_idx = i % num_cols +1
        # Extract true labels and predicted probabilities for the current class
        true_labels = targets[class_label]
        predicted_probs_class = predicted_probs[class_label]

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs_class)
        roc_auc = auc(fpr, tpr)
        opt_idx=np.argmax(tpr-fpr)
        opt_thresh = thresholds[opt_idx]
        opt_thresholds.append(opt_thresh)
        
        # Plot ROC curve for the current class
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                        line=dict(color='darkorange', width=2),
                        name=f'ROC curve (area = {roc_auc:.2f})'),
                        row=row_idx, col=col_idx)
        
        # Add straight line axes
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                        line=dict(color='navy', width=2, dash='dash'),
                        showlegend=False),
                        row=row_idx, col=col_idx)
        
        #Set axis labels for the second row
        if row_idx == 2:
            fig.update_xaxes(title_text='False Positive Rate', row=row_idx, col=col_idx)
        # Set axis labels for the first column
        if col_idx == 1:
            fig.update_yaxes(title_text='True Positive Rate', row=row_idx, col=col_idx)
        
        # Add annotation with ROC AUC value to the bottom right corner
        fig.add_annotation(
            x=0.8,
            y=0,
            xref='paper',
            yref='paper',
            text=f'AUC = {roc_auc:.2f}',
            showarrow=False,
            font=dict(size=10),
            row=row_idx,
            col=col_idx
        )
       
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curves',
        showlegend=False,
    )
    #plt.tight_layout()

    return fig, opt_thresholds
