#image encoder pretraining script

import torch
from tqdm import tqdm
import os 
import logging
import time
import wandb
import copy  
import time
import sys 

sys.path.insert(0,  os.path.abspath(".."))
from utils.utils import * 
from utils.run_configurations import *
from utils.path_datasets_and_weights import *
from utils.test_utils import test_main

### SETUP ###
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

if CONTINUE_TRAINING == False:
    wandb_name = f"Run {RUN}"
else:
    wandb_name = f"Run {RUN} cont."

wandb_log = False #True
if wandb_log:
    wandb.init(
        # set the wandb project where this run will be logged
        project= "GenX ImEnc", 
        name = wandb_name,
        # track hyperparameters and run metadata
        config={
        "RUN": RUN,
        "RUN_COMMENT": RUN_COMMENT,
        "PRETRAIN_LLM": PRETRAIN_LLM,
        "PRETRAIN_CNN": PRETRAIN_CNN,
        "TRAIN_FULL_MODEL": TRAIN_FULL_MODEL,
        "dataset": "MIMIC-CXR",
        "epochs": NUM_EPOCHS,
        }
    )

#device
cuda_device_to_see = CUDA_DEVICE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.cuda.is_available())
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
print(device.type)


### TRAINING FUNCTION ###
def train_im_model(model, train_dl, val_dl, epochs, checkpoints, log_file, start_epoch):
    """
    Training loop for the image model.
    """
    run_params = train_setup(model,epochs, checkpoints, device, log)
    criterion = run_params["IM_LOSS"]
    optimizer = run_params["IM_OPTIMISER"]
    early_stopping_count = 0
    highest_f1 = 0
    highest_bal_acc = 0

    for epoch in tqdm(range(epochs)):
        log.info(f"Training epoch {epoch}!")
        if epoch == 0:
            start_time = time.time()
        train_targets = []
        train_outputs = []

        for num_batch, batch in tqdm(enumerate(train_dl)):
            images, targets = batch.values()
            model.train()
            optimizer.zero_grad()
            feat, output = model(images.to(device))
            if NO_IM_MODEL_CLASSES == 1:
                targets = targets.unsqueeze(-1)
            loss = criterion(output, targets.float().to(device)) 
            loss.backward()
            optimizer.step()
            loss=loss.item()
            if wandb_log:
                wandb.log({"train loss": loss})
            
            if PRETRAIN_CNN:
                train_targets.append(copy.deepcopy(targets))
                train_outputs.append(copy.deepcopy((output.detach().to('cpu')>0.5).float()))
                del targets, output

            #validation loop
            if (num_batch%EVALUATE_EVERY_K_BATCHES == 0) and (num_batch>0):
                #logging
                log.info(num_batch)
                train_metrics = metrics(train_targets, train_outputs)
                f1, bal_acc, val_metrics, val_loss = val_im_model(model, run_params, val_dl, num_batch)
                if wandb_log:
                    log_metrics(val_metrics, train_metrics)
                
                train_targets = []
                train_outputs = []

                if wandb_log:
                    wandb.log({"LR": float(optimizer.param_groups[0]["lr"])})

                #early stopping
                highest_f1, highest_bal_acc,early_stopping_count = early_stopping(log, f1, highest_f1, bal_acc, highest_bal_acc, run_params, epoch, val_loss, loss, optimizer, model, early_stopping_count, start_epoch)
            
            torch.cuda.empty_cache()

        #save final checkpoint as well as "best"
        save_model(run_params, epoch, val_loss, optimizer,model, log, final=True)
        if epoch == 0:
            end_time = time.time()
            time_taken = end_time - start_time
            log.info(f"one epoch took {time_taken/60} mins")
        log.info(f"Epoch: {epoch+1}, Train Loss: {loss}, Val loss: {val_loss}")
    return 

### VALIDATION FUNCTION ###
@torch.no_grad()       
def val_im_model(model, run_params, val_dl, num_batch):
    """
    Validation loop for the image model.
    """
    model.eval()
    val_targets = []
    val_outputs = []
    val_losses = []
    val_losses_individual = []
    val_prob_outputs = []
    counter=0
    for no, batch in tqdm(enumerate(val_dl)):
        images, targets = batch.values()
        feat, output = model(images.to(device))

        if NO_IM_MODEL_CLASSES == 1:
            targets = targets.unsqueeze(-1)
        val_targets.append(targets)
        val_loss = run_params["IM_LOSS"](output, targets.float().to(device))
        val_losses.append(val_loss.item())
        indiv_losses = run_params["IM_INDIV_LOSS"](output, targets.float().to(device))
        val_losses_individual.append(torch.mean(indiv_losses.detach().to('cpu'), dim=0))
        prob_outputs = output.detach().to('cpu')
        val_prob_outputs.append(prob_outputs)
        val_outputs.append((prob_outputs>0.5).float())
    
    val_metrics, val_targets, val_outputs, val_prob_outputs, roc_curves = metrics(val_targets, val_outputs, return_data = True, prob_outputs=val_prob_outputs)
    
    #loss per disease
    val_losses_individual = indiv_losses2(val_losses_individual, log)

    if wandb_log:
        wandb.log({"val loss": np.mean(val_losses)})
    if counter ==10:
        #just to reduce logging freq as wandb not happy
        if wandb_log:
            wandb.log(val_losses_individual)
        #wandb.log({f"roc_curves/val":roc_curves}) #if you want to log roc curves
        counter=0
    else:
        counter =+1

    f1 = val_metrics.loc['f1'].mean()
    bal_acc = val_metrics.loc['bal_accuracy'].mean()

    return f1, bal_acc, val_metrics, np.mean(val_losses)#, val_loss_individual

def log_metrics(val_metrics, train_metrics):
    metrics_rows = ['accuracy','precision','recall','f1','bal_accuracy']
    for metric in metrics_rows:
        wandb.log({f'Val Metrics/{metric}': val_metrics.loc[metric].mean(), f'Train Metrics/{metric}': train_metrics.loc[metric].mean()}, commit=False)
    wandb.log({f'Val Metrics/auc': val_metrics.loc['auc'].mean()})
    return

def dataloaders(train_dataset, val_dataset, tokenizer =None):
    if PRETRAIN_CNN:
        im_dataloaders = get_dataloaders(train_dataset, val_dataset,log)
        return im_dataloaders
    elif PRETRAIN_LLM:
        llm_dataloaders = 0
        return llm_dataloaders
    elif TRAIN_FULL_MODEL:
        full_dataloaders = get_dataloaders(train_dataset, val_dataset,log, tokenizer=tokenizer)
        return full_dataloaders


def main():
    print("Starting training")
    print(f"no image model classes: {NO_IM_MODEL_CLASSES}")
    ### SETUP ###
    if PRETRAIN_CNN:
        path_runs = path_runs_image_model
        (checkpoints_folder_path, 
        config_file_path, generated_sentences_and_reports_folder_path, 
        log_file, run_folder_path) = create_run_folder(log, path_runs)
    start_epoch = 0
    if CONTINUE_TRAINING and PRETRAIN_CNN:
        im_checkpoint, last_saved_epoch= get_checkpoint(RUN)
        model = get_im_model(device, checkpoint=im_checkpoint)
        log.info(f"Continuing to train run {RUN} from epoch {last_saved_epoch}.")
        start_epoch = last_saved_epoch
    elif PRETRAIN_CNN:
        model = get_im_model(device)

    ### MODEL TRAINING ###
    if PRETRAIN_CNN:
        train_dataset, val_dataset = get_datasets(path_full_dataset_im, config_file_path, log)
        train_loader, val_loader = dataloaders(train_dataset, val_dataset)
        log.info(f"length of the dataloaders: {len(train_loader)}, {len(val_loader)}")
    
        log.info("Training image model")
        start_time = time.time()
        train_im_model(
            model=model,
            train_dl=train_loader,
            val_dl=val_loader,
            epochs=NUM_EPOCHS,
            checkpoints=checkpoints_folder_path,
            log_file = log_file,
            start_epoch = start_epoch
        )

        log.info("Finished Training.")
        end_time = time.time()
        elapsed_time = end_time - start_time
        log.info(f'training time: {elapsed_time/(60*60)} hrs')

    ### MODEL TESTING ###
    if TEST_CNN:
        log.info("Testing")
        start_time = time.time()
        test_metrics, conf_mat, roc_curves = test_main(device, log, test=True)

        if wandb_log:
            wandb.log({"test/confusion matrix": wandb.Image(conf_mat, caption="Confusion matrices")}, commit =False)
            wandb.log({"test/metrics": wandb.Table(data=test_metrics, columns=test_metrics.columns)}, commit =False)
            wandb.log({f"test/roc_curves":roc_curves})

        log.info("Finished Testing.")
        end_time = time.time()
        elapsed_time = end_time - start_time
        log.info(f'testingtime: {elapsed_time/(60*60)} hrs')

    if wandb_log:
        wandb.finish()

if __name__ == "__main__":
    main()
