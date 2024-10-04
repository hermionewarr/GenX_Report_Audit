import pandas as pd
import os
import sys
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import time 
import copy
import math
from dataclasses import dataclass, asdict

sys.path.insert(0,  os.path.abspath(".."))
#print(sys.path)
from utils.utils import CustomLLM_Dataset, get_tokenized_datasets, get_torch_transforms, get_dataloaders, get_checkpoint, create_run_folder, get_tokenizer, findings_or_impression
from utils.path_datasets_and_weights import path_full_dataset_full, path_runs_full_model
from utils.run_configurations import *
from full_model.model.GenX import GenX, GenXConfig
from vision_model.im_encoder import ImageEncoder
from full_model.test_and_QC import test_main, get_all_metrics
from datasets import Dataset

# wandb logging
wandb_log = False #True
wandb_project = 'GenX Full Model'
wandb_run_name = f'Run {RUN}' 

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
torch.set_float32_matmul_precision("high") 
#adamw optimizer
learning_rate = LR #6e-4 # max learning rate
max_iters = 1000 # total number of training iterations 
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = False #True # whether to decay the learning rate dont end up using this
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 200000 #1000 # should be ~= max_iters per Chinchilla
min_lr = learning_rate/10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla


if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name)
    
def load_im_model(im_run, train=UPDATE_IMAGE_MODEL):
    im_checkpoint,_ = get_checkpoint(im_run)
    #ImEncoder = ImageEncoder(return_feature_vectors=True)
    ImEncoder = ImageEncoder(return_feature_vectors=False) # return classifier outputs and feature vectors
    ImEncoder.to(device, non_blocking=False)
    if train:
        log.info("Updating image model.")
        if CONTINUE_TRAINING:
            log.info("Loading image model.")
            im_model_path = f"/home/hermione/Documents/VLP/GenX/src/runs/full_model/run_{RUN}/checkpoints/ImEncoder.pt"
            checkpoint = torch.load(im_model_path)
            ImEncoder.load_state_dict(checkpoint["model"])
        else:
            ImEncoder.load_state_dict(torch.load(im_checkpoint)["model"])
        ImEncoder.train()
    else:
        ImEncoder.load_state_dict(torch.load(im_checkpoint)["model"])
        ImEncoder.eval()
        for param in ImEncoder.parameters():
            param.requires_grad = False

    return ImEncoder

def get_datasets():
    usecols = [
        "mimic_image_file_path",
        "labels",
        "image_id",
        findings_or_impression()
    ]

    datasets_as_dfs = {}
    datasets_as_dfs["train"] = pd.read_csv(os.path.join(path_full_dataset_full, FULL_DATASET_NAME), usecols=usecols) 
    datasets_as_dfs["valid"] = pd.read_csv(os.path.join(path_full_dataset_full, FULL_VAL_DATASET_NAME), usecols=usecols) 


    log.info(len(datasets_as_dfs["train"]))
    datasets_as_dfs["train"] = datasets_as_dfs["train"][datasets_as_dfs["train"][findings_or_impression()].notna() & datasets_as_dfs["train"]["labels"].notna()]
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][datasets_as_dfs["valid"][findings_or_impression()].notna() & datasets_as_dfs["valid"]["labels"].notna()]
    log.info(len(datasets_as_dfs["train"]))

    log.info(f'val data length {len(datasets_as_dfs["valid"])}, train data length {len(datasets_as_dfs["train"])}')
    
    raw_train_dataset = Dataset.from_pandas(datasets_as_dfs["train"])
    raw_val_dataset = Dataset.from_pandas(datasets_as_dfs["valid"])
    # tokenize the raw datasets
    tokenized_train_dataset, tokenized_val_dataset = get_tokenized_datasets(raw_train_dataset, raw_val_dataset)
    train_dataset = CustomLLM_Dataset(tokenized_train_dataset, log, transforms=get_torch_transforms("train"))
    val_dataset = CustomLLM_Dataset(tokenized_val_dataset, log, transforms=get_torch_transforms("val"))

    return train_dataset, val_dataset

def setup(no_im_tokens, im_train=False):
    print("no im tokens: ", no_im_tokens)
    model = GenX(GenXConfig(), no_im_tokens).to(device)
    model = torch.compile(model)
    lr = learning_rate
    train_dataset, val_dataset = get_datasets()
    train_dl, val_dl = get_dataloaders(train_dataset, val_dataset, log)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2))
    if CONTINUE_TRAINING:
        model_path = f"/home/hermione/Documents/VLP/GenX/src/runs/full_model/run_{RUN}/checkpoints/GenX.pt"
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model"])
        lr = checkpoint["lr"] 
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        iter_num = checkpoint["iter_num"]
        
    else:
        
        start_epoch = 0
        iter_num = 0
    
    model.train()

    IM_RUN = 32 #36 #32
    ImageEncoder = load_im_model(IM_RUN, train=train)
    im_optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    (checkpoints_folder_path, config_file_path, generated_sentences_and_reports_folder_path, 
     log_file, run_folder_path) = create_run_folder(log, path_runs_full_model)
    
    model_config = asdict(GenXConfig())
    with open(config_file_path, "a") as f:
        f.write("\nconfig = " + str(model_config))

    if wandb_log:
        log.info(f"cwd: , os.getcwd()")
        model_artifact = wandb.Artifact("GenX", type="model")
        model_artifact.add_dir(".")
        wandb.run.log_artifact(model_artifact)
    return model, train_dl, val_dl, optimizer, ImageEncoder, checkpoints_folder_path, generated_sentences_and_reports_folder_path, im_optimiser, start_epoch, iter_num

def train(model, ImEncoder, epoch, iter_num, train_dl, optimizer, val_dl, im_optimiser, checkpoints_folder_path, generated_sentences_and_reports_folder_path, best_f1, best_val_loss):
    train_loss = []
    model.train()
    log.info(f"iter_num: {iter_num}")
    for no, batch in tqdm(enumerate(train_dl)):
        images  = batch["image"]
        features,_ = ImEncoder(images.to(device, non_blocking=True)) #[B,512]
        #im_classes = (classification_prob >0.5).float()
        
        im_lab = batch["labels"]
        input_ids = batch["input_ids"]
        
        attn_mask = batch["attention_mask"]

        #Update lr if needed
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        if UPDATE_IMAGE_MODEL:
            im_optimiser.zero_grad() #if trianing the image model too
        logits, loss = model(input_ids.to(device, non_blocking=True), features.to(device, non_blocking=True), attn_mask.to(device, non_blocking=True), targets=input_ids.to(device, non_blocking=True)) #added no blocking =true to try and speed up stuff
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        #if training the image model too
        if UPDATE_IMAGE_MODEL:
            #features.backward(loss.grad)
            im_optimiser.step()

        if no % EVALUATE_EVERY_K_BATCHES == 0:
            if no == 0 and (epoch != 0) and (epoch%2 == 0): #this loop slows it down a bit.
                val_loss, f1 = val(model, ImEncoder, epoch, val_dl, generated_sentences_and_reports_folder_path, log_metric=True)
                best_f1, best_val_loss = save_best_model(model, val_loss, f1, checkpoints_folder_path, best_f1, best_val_loss, epoch, optimizer)
                log.info(f'best f1: {best_f1}, best val loss: {best_val_loss}')
            else:
                val_loss= val(model, ImEncoder, epoch, val_dl,generated_sentences_and_reports_folder_path)
            t_loss = sum(train_loss)/len(train_loss)
            log.info(f'epoch: {epoch}, batch: {no}, train loss: {t_loss:.3g}, val loss: {val_loss:.3g}')
            if wandb_log:
                wandb.log({"train_loss": t_loss, "val_loss": val_loss, "lr": lr})
        torch.cuda.empty_cache()
        iter_num += 1
        
    train_loss = sum(train_loss)/len(train_loss)
    log.info(f'epoch {epoch} train_loss {train_loss}')
    if UPDATE_IMAGE_MODEL:
        checkpoint = {"model": ImEncoder.state_dict()}
        torch.save(checkpoint, checkpoints_folder_path + "/ImEncoder.pt")
    return train_loss, model, best_f1, best_val_loss, iter_num, lr

@torch.no_grad()
def val(model, ImEncoder, epoch, val_dl, generated_sentences_and_reports_folder_path, log_metric=False):
    tokeniser = get_tokenizer()
    val_loss = []
    model.eval()
    val_ref_reports = []
    val_gen_reports = []
    ImEncoder.eval()
    for batch in val_dl: 
        #alternativly do one batch but generete it instead.
        images  = batch["image"]
        features,_ = ImEncoder(images.to(device)) #[2,512]
        im_lab = batch["labels"]
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        reference_reports = batch["reference_reports"]
        
        logits, loss = model(input_ids.to(device), features.to(device), attn_mask.to(device), targets=input_ids.to(device))
        val_loss.append(loss.item())
        max_index = torch.argmax(logits, dim=-1)
        
        reports = tokeniser.batch_decode(max_index,skip_special_tokens=True)
        del loss, logits, max_index

        val_gen_reports.append(copy.deepcopy(reports))
        val_ref_reports.append(copy.deepcopy(reference_reports))
        torch.cuda.empty_cache()
    
    model.train()
    if UPDATE_IMAGE_MODEL:
        ImEncoder.train()
    val_loss = sum(val_loss)/len(val_loss)

        #we need to generate some reports and get the NLP metrics and CE metrics
    if log_metric:
        ref_reports = [item for sublist in val_ref_reports for item in sublist] #torch.cat(ref_reports, dim=0).tolist()
        gen_reports = [item for sublist in val_gen_reports for item in sublist]
        report_df = pd.DataFrame({"ref_reports": ref_reports, "gen_reports": gen_reports})
        del val_ref_reports
        del val_gen_reports
        all_metrics = get_all_metrics(report_df, RUN, generated_sentences_and_reports_folder_path, epoch = None, single_disease_class=False, val = True)
        all_metrics = all_metrics.to_dict()["metrics"]
        log.info(all_metrics)
        if wandb: 
            nlp_metrics = ["bleu_1","bleu_2","bleu_3", "bleu_4", "rouge", "meteor"]
            ce_metrics = ["accuracy", "precision", "recall", "f1", "bal_accuracy"]
            for metric in nlp_metrics:
                wandb.log({f"NLP Metrics/{metric}" : all_metrics[metric]}, commit=False)
            for i, metric in enumerate(ce_metrics):
                commit = False if i < (len(ce_metrics) - 1) else True
                wandb.log({f"CE Metrics/{metric}" : all_metrics[metric]}, commit = commit)
        f1 = all_metrics["f1"]
        return val_loss, f1
    else:
        return val_loss

def save_best_model(model, val_loss, f1, checkpoints_folder_path, best_f1, best_val_loss, epoch, optimizer):
    log.info(f"loss and f1: {best_val_loss}, {best_f1}, {val_loss}, {f1}")
    checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "f1": f1
        }
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(checkpoint, checkpoints_folder_path + "/GenX_val.pt")
        log.info(f"model val loss improved at epoch {epoch}, saved to {checkpoints_folder_path + '/GenX_val.pt'}")
    if f1 > best_f1:
        best_f1 = f1
        for f in os.listdir(checkpoints_folder_path):
            if f.startswith('GenX_f1'):
                os.remove(os.path.join(checkpoints_folder_path,f))
        torch.save(checkpoint, checkpoints_folder_path + f"/GenX_f1_e{epoch}.pt")
        log.info(f"model f1 improved at epoch {epoch}, saved to {checkpoints_folder_path + '/GenX_f1.pt'}")
        

    return best_f1, best_val_loss

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    if decay_lr:
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
    else:
        return LR

def main():
    im_train=UPDATE_IMAGE_MODEL
    no_im_tokens=NO_IMAGE_TOKENS 
    # initialising best values
    best_val_loss = 20
    best_f1 = 0

    (model, train_dl, val_dl, optimizer, ImEncoder, 
    checkpoints_folder_path, generated_sentences_and_reports_folder_path, 
    im_optimiser, start_epoch, iter_num) = setup(no_im_tokens=no_im_tokens, im_train=im_train)

    start_time = time.time()
    #iter_num =0
    #training
    log.info(f"training model for {FULL_MODEL_EPOCHS} epochs")
    for epoch in range(start_epoch+1, FULL_MODEL_EPOCHS+1):

        train_loss, model, best_f1, best_val_loss, iter_num, lr  = train(model, ImEncoder, epoch, iter_num, train_dl, optimizer, val_dl, im_optimiser, checkpoints_folder_path, generated_sentences_and_reports_folder_path, best_f1, best_val_loss)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "lr": lr,
            "iter_num": iter_num
        }
        torch.save(checkpoint, checkpoints_folder_path + "/GenX.pt")
        if (epoch%10 == 0): #starts from 0
            log.info(f"saving model at epoch {epoch}.")
            torch.save(checkpoint, checkpoints_folder_path + f"/GenX_e{epoch}.pt")
        

        if epoch ==1:
            end_time = time.time()
            elapsed_time = end_time - start_time
            log.info(f'one epoch took: {elapsed_time/(60*60)} hrs')

    end_time = time.time()
    elapsed_time = end_time - start_time
    log.info(f'training time: {elapsed_time/(60*60)} hrs')

    #testing
    test_main(RUN,no_im_tokens=no_im_tokens, im_mod_retrain = im_train)
    #need to run test on best f1 too

    return

if __name__ == "__main__":
    main()