"""
script to run the full model test set.
takes about 15 mins to run

Set which run of the full model you want to use in main(): FULL_RUN

QC pipeline here
step 1: Inference: Generate reports from the test set. 
step 2: Label the Images using auxiliay ML models. Our ACs are single disease classifiers (resnet-18s).
step 3: Label the reports using the chexpert labeler.
step 4: Evaluation: compare the generated report labels with the reference report labels.
step 5: Quality Control: compare the generated report labels with the image labels. ce_metrics_disease_match()
"""
import torch
import os
import tempfile
import csv
import re
import pandas as pd
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import numpy as np
import copy
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import ast 

import sys
sys.path.insert(0,  os.path.abspath(".."))
from CheXbert.src.constants import CONDITIONS
from CheXbert.src.label import label, save_preds
from CheXbert.src.models.bert_labeler import bert_labeler

from utils.path_datasets_and_weights import path_chexbert_weights, path_full_dataset_full
from utils.utils import get_checkpoint, get_torch_transforms, get_dataloaders, get_tokenizer, findings_or_impression
from utils.custom_dataset import CustomLLM_Dataset
from utils.custom_collator import CustomCollator
from utils.run_configurations import *
from vision_model.im_encoder import ImageEncoder
from full_model.model.GenX import GenX, GenXConfig
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'GenX Full Model'
wandb_run_name = f'Test of R{RUN}' 
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name)

### IMAGE MODEL LOAD ###
IM_RUN = 32 #-which ever pretrained run you want to use

def load_im_model(im_run, full_run, im_mod_retrain=False):
    print(f"loading image model {im_run}.")
    im_checkpoint,_ = get_checkpoint(im_run)
    #ImEncoder = ImageEncoder(return_feature_vectors=True)
    ImEncoder = ImageEncoder(return_feature_vectors=False)
    ImEncoder.to(device, non_blocking=False)
    if im_mod_retrain:
        if CONTINUE_TRAINING:
            print("loading cont retrained im model")
            im_checkpoint = f"/home/hermione/Documents/VLP/GenX/src/runs/full_model/run_{full_run}/cont/checkpoints/ImEncoder.pt"
            ImEncoder.load_state_dict(torch.load(im_checkpoint)["model"])
        else:
            print("loading retrained image model")
            im_checkpoint = f"/home/hermione/Documents/VLP/GenX/src/runs/full_model/run_{full_run}/checkpoints/ImEncoder.pt"
            ImEncoder.load_state_dict(torch.load(im_checkpoint)["model"])
    else:
        ImEncoder.load_state_dict(torch.load(im_checkpoint)["model"])
    ImEncoder.eval()
    #ImageEncoder = get_im_model(device, im_checkpoint)
    for param in ImEncoder.parameters():
        param.requires_grad = False

    return ImEncoder

def load_single_disease_im_model(im_run):
    print("loading single disease classifier")
    im_checkpoint,_ = get_checkpoint(im_run)
    SDImEncoder = ImageEncoder(return_feature_vectors=False, single_disease = True)
    SDImEncoder.to(device, non_blocking=False)
    SDImEncoder.load_state_dict(torch.load(im_checkpoint)["model"])
    SDImEncoder.eval()
    #ImageEncoder = get_im_model(device, im_checkpoint)
    for param in SDImEncoder.parameters():
        param.requires_grad = False
    return SDImEncoder

#label the reports using the chexpert labeler
def get_chexbert():
    model = bert_labeler()
    #model = nn.DataParallel(model)  # needed since weights were saved with nn.DataParallel
    checkpoint = torch.load(path_chexbert_weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    return model

def get_chexbert_labels_for_gen_and_ref_reports(chexbert, report_df, run, epoch, testing= False):
    reports_list =[[gen_report] for gen_report in report_df["gen_reports"]]
    print("size check: ", len(reports_list))
    print(reports_list[:2])
    
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_gen_reports_file_path = os.path.join(temp_dir, "gen_reports.csv")
        csv_ref_reports_file_path = os.path.join(temp_dir, "ref_reports.csv")

        header = ["Report Impression"]

        with open(csv_gen_reports_file_path, "w") as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(header)
            csv_writer.writerows([[gen_report] for gen_report in report_df["gen_reports"]])

        with open(csv_ref_reports_file_path, "w") as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(header)
            csv_writer.writerows([[ref_report] for ref_report in report_df["ref_reports"]])

        
        preds_gen_reports = label(path_chexbert_weights, csv_gen_reports_file_path)
        preds_ref_reports = label(path_chexbert_weights, csv_ref_reports_file_path)
        #disease order: Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices,No Finding
       
        path = save_path(run, epoch, testing=testing)
        df = save_preds(preds_gen_reports, csv_gen_reports_file_path, path, "gen")
        df2 = save_preds(preds_ref_reports, csv_ref_reports_file_path, path, "ref")
    #print(df[:2])
    return preds_gen_reports, preds_ref_reports, df, df2


def get_one_dataset():
    usecols = [
        "mimic_image_file_path",
        "labels",
        "image_id",
        findings_or_impression()
    ]
    
    datasets_as_dfs = {}
    datasets_as_dfs = pd.read_csv(os.path.join(path_full_dataset_full, FULL_TEST_DATASET_NAME), usecols=usecols) 

    log.info(len(datasets_as_dfs)) #3744
    datasets_as_dfs = datasets_as_dfs[datasets_as_dfs[findings_or_impression()].notna() & datasets_as_dfs["labels"].notna()]
    log.info(len(datasets_as_dfs)) #2607
    
    raw_dataset = Dataset.from_pandas(datasets_as_dfs)

    # tokenize the raw datasets
    tokenized_dataset = get_tokenized_dataset(raw_dataset)
    dataset = CustomLLM_Dataset(tokenized_dataset, log, transforms=get_torch_transforms("test"))

    return dataset

def get_tokenized_dataset(raw_dataset):
    tokenizer = get_tokenizer()
    def tokenize_function(example):
        phrases = example[findings_or_impression()]  # (str)  - make List[str]?
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        phrases_with_special_tokens = [bos_token + phrases + eos_token]

        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)#1024

    tokenized_dataset = raw_dataset.map(tokenize_function)
    return tokenized_dataset

def setup(FULL_RUN,tokenizer, epoch =None,  no_im_tokens =1, im_mod_retrain=False, testing=False):
    dataset = get_one_dataset()
    dl = DataLoader(
            dataset,
            collate_fn=CustomCollator(tokenizer),
            batch_size= 8, #BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS, 
            pin_memory=False,
            persistent_workers=False,
        )
    model = GenX(GenXConfig(),no_im_tokens).to(device, non_blocking=False)
    model = torch.compile(model) 
    
    print(f"loading run {FULL_RUN} model.")
    if CONTINUE_TRAINING:
        model_path = f"/src/runs/full_model/run_{FULL_RUN}/cont/checkpoints/GenX.pt"
    else:
        if epoch ==None:
            model_path = f"/src/runs/full_model/run_{FULL_RUN}/checkpoints/GenX.pt"
        else:
            model_path = f"/src/runs/full_model/run_{FULL_RUN}/checkpoints/GenX_e{epoch}.pt"
    model.load_state_dict(torch.load(model_path)["model"])
   
    ImageEncoder = load_im_model(IM_RUN, FULL_RUN, im_mod_retrain)
    
    return model, dl, ImageEncoder

def test(test_dl, model, tokeniser, ImEncoder, im_mod_retrain, run, path, no_im_tokens, one_class = False, testing=False, single_disease_class = False):
    ref_reports = []
    gen_reports = []
    disease_labels = []
    ids = []
    ImEncClasses = []
    ImClassProb = []
    SDImClassProb = []
    SDImClass = []

    model.eval()
    if single_disease_class:
        print(f"single disease: {SD_DISEASE}")
        SDImEnc = load_single_disease_im_model(im_run = SD_RUN)
    if im_mod_retrain:
        AC_14 = load_im_model(IM_RUN, FULL_RUN, im_mod_retrain =False) #so we get the pretrained encoder we want to use
    else:
        AC_14 = ImEncoder
    for no, batch in tqdm(enumerate(test_dl)):
        # get the inputs
        inputs = batch["input_ids"].to(device)
        #attention_mask = batch["attention_mask"].to(device)
        id = batch["mimic_image_file_path"]
        labels = batch["labels"]
        images = batch["image"]
        reference_reports = batch["reference_reports"]

        disease_labels.append(copy.deepcopy(labels))
        ref_reports.append(copy.deepcopy(reference_reports))
        ids.append(copy.deepcopy(id))

        inputs = inputs[:,:1] #just feed in bos token

        features, classification_prob  = ImEncoder(images.to(device))
        _, classification_prob  = AC_14(images.to(device))
        #if want to inestigate using the image encoder as ACs
        ImClassProb.append(copy.deepcopy(classification_prob))
        im_classes = (classification_prob >0.5).float()
        ImEncClasses.append(copy.deepcopy(im_classes)) #diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema','Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
                                        #'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax','Support Devices']

        #generate reports
        outputs = model.generate(im_feats=features, max_new_tokens =212, idx =inputs)#, no_im_tokens=no_im_tokens
        generated_reports = tokeniser.batch_decode(outputs, skip_special_tokens=True)
        
        gen_reports.append(copy.deepcopy(generated_reports))

        if single_disease_class:
            #using single disease classifier to label images with disease x present or absent
            _, classification_prob  = SDImEnc(images.to(device))
            SDImClassProb.append(copy.deepcopy(classification_prob))
            SDImClass.append(copy.deepcopy((classification_prob >0.5).float()))
            
        del outputs
        del inputs
        del labels
        del images
        del reference_reports

        if wandb_log:
            wandb.log({"batch no": no})
        torch.cuda.empty_cache()
    ref_reports = [item for sublist in ref_reports for item in sublist] 
    gen_reports = [item for sublist in gen_reports for item in sublist]
    disease_labels = [str(item.tolist()) for sublist in disease_labels for item in sublist]
    ids = [item for sublist in ids for item in sublist]
    ImEncClasses = [str(item.tolist()) for sublist in ImEncClasses for item in sublist]
    ImClassProb = [str(item.tolist()) for sublist in ImClassProb for item in sublist]

    report_df = pd.DataFrame({"ref_reports": ref_reports, "gen_reports": gen_reports, "disease_labels": disease_labels, "ids": ids})
    report_df_classes = pd.DataFrame({"ref_reports": ref_reports, "gen_reports": gen_reports, "disease_labels": disease_labels, "ids": ids, "ImEncClasses": ImEncClasses, "ClassProb":ImClassProb})
    
    if single_disease_class:
        SDImClassProb = [str(item.tolist()) for sublist in SDImClassProb for item in sublist]
        SDImClass = [str(item.tolist()) for sublist in SDImClass for item in sublist]
        report_df_classes = pd.DataFrame({"ref_reports": ref_reports, "gen_reports": gen_reports, "disease_labels": disease_labels, "ids": ids, "ImEncClasses": ImEncClasses, "ClassProb":ImClassProb, "SDImClasses": SDImClass, "SDImClassProb": SDImClassProb})
    if testing == False:
        report_df.to_csv(path +"/test_reports.csv", index=False)
        report_df_classes.to_csv(path +"/test_ImClasses.csv", index=False)
        if single_disease_class:
            report_df_classes.to_csv(path +f"/{SD_DISEASE}_ImClasses.csv", index=False)
    return report_df_classes

def save_path(run, epoch =None, testing=False):
    if CONTINUE_TRAINING:
        save_path= f"/src/runs/full_model/run_{run}/cont/generated_sentences_and_reports/generated_reports"
    else:
        if epoch==None:
            save_path = f"/src/runs/full_model/run_{run}/generated_sentences_and_reports/generated_reports"
        else:
            save_path = f"/src/runs/full_model/run_{run}/generated_sentences_and_reports_e{epoch}/generated_reports"
    if testing:
        save_path = "/src/full_model"
    return save_path

def labeler14U(diseases, df):
    df['labels'] = df[diseases].apply(lambda row: [1 if (row[disease] == 1 or row[disease] ==-1) else 0 for disease in diseases], axis=1)
    df = df.drop(columns=diseases)
    return df

def labeler14_keepU(diseases, df):
    #if want to use the uncertainty labels
    df['labels'] = df[diseases].apply(lambda row: [1 if (row[disease] == 1) else -1 if (row[disease] ==-1) else 0 for disease in diseases], axis=1)
    df = df.drop(columns=diseases)
    return df

def convert_labels(results_df, diseases):
    outputs = list(map(list, zip(*[lab for lab in results_df["Gen Labels"]]))) 
    targets = list(map(list, zip(*[lab for lab in results_df["Ref Labels"]])))
    #labels = list(map(list, zip(*[lab for lab in results_df["disease_labels"]])))
    df_outputs = pd.DataFrame(outputs, index=diseases).T
    df_targets = pd.DataFrame(targets, index=diseases).T
    #df_labels = pd.DataFrame(labels, index=diseases).T
    return df_outputs, df_targets #, df_labels

def ce_metrics(df_targets, df_outputs, diseases, path, chex, testing=False, val=False):
    #print(diseases)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    bal_accuracy = []
    auc = []
    five_acc = []
    print("print targets and outputs")
    five_diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema','Pleural Effusion'] # 5 diseases used in chexpert competition.
    for disease in diseases:
        if sum(df_outputs[disease])==0:
            print("No reports with this disease.")
            accuracy.append(accuracy_score(df_targets[disease], df_outputs[disease]))
            precision.append(np.nan)
            recall.append(np.nan)
            f1.append(np.nan)
            bal_accuracy.append(balanced_accuracy_score(df_targets[disease], df_outputs[disease]))
        else:
            acc = accuracy_score(df_targets[disease], df_outputs[disease])
            accuracy.append(acc)
            precision.append(precision_score(df_targets[disease], df_outputs[disease], zero_division=1))
            recall.append(recall_score(df_targets[disease], df_outputs[disease], zero_division=1))
            f1.append(f1_score(df_targets[disease], df_outputs[disease], zero_division=1))
            bal_accuracy.append(balanced_accuracy_score(df_targets[disease], df_outputs[disease]))
        #print(disease)
        #print(f"F1: {f1[-1]}")

        if disease in five_diseases:
            five_acc.append(accuracy_score(df_targets[disease], df_outputs[disease]))
            #print("acc", five_acc[-1])
        
        
    if val == False:
        #using sklearns precision_recall_fscore_support to comapre with tanida and muira
        p_sup,r_sup,f1_sup,_ = precision_recall_fscore_support(df_targets, df_outputs, average='macro')
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(df_targets, df_outputs, average='micro')
        p_sup_five,r_sup_five,f1_sup_five,_ = precision_recall_fscore_support(df_targets[five_diseases], df_outputs[five_diseases], average='macro')
        p_micro_five, r_micro_five, f1_micro_five, _ = precision_recall_fscore_support(df_targets[five_diseases], df_outputs[five_diseases], average='micro')
        #f1_sup_cal.append(f1_sup)
        print("marco:")
        print(p_sup,r_sup,f1_sup)
        print("micro:")
        print(p_micro, r_micro, f1_micro)
        print("marco 5:")
        print(p_sup_five,r_sup_five,f1_sup_five)
        print("micro 5:")
        print(p_micro_five, r_micro_five, f1_micro_five)
        print("5 accuracy:", np.mean(five_acc))
        acc_5_mirco = accuracy_score(df_targets[five_diseases].values.flatten(), df_outputs[five_diseases].values.flatten())
        acc_14_micro = accuracy_score(df_targets.values.flatten(), df_outputs.values.flatten())
        print("accuracy 5/ 14 micro: ", acc_5_mirco, acc_14_micro) #yeah it is the same as the other  accuracy

    metrics = np.vstack((accuracy, precision, recall, f1, bal_accuracy))
    df_metrics = pd.DataFrame(metrics, columns=diseases, index = ['accuracy','precision','recall','f1','bal_accuracy'])
    #get the means of the metrics
    mean_ce = df_metrics.mean(axis=1)
    print("og metrics")
    print(mean_ce)
    if testing == False:
        df_metrics.to_csv(f"{path}/generated_reports/{chex}_clinical_metrics.csv")
        mean_ce.to_csv(f"{path}/generated_reports/{chex}_av_clinical_metrics.csv")
    else:
        df_metrics = df_metrics.round(3)
        df_metrics.to_csv(f"/src/full_model/{chex}_clinical_metrics.csv", index=False)
    mean_ce = pd.DataFrame(mean_ce, columns=['metrics'])
    return df_metrics, mean_ce

def ce_metrics_disease_match(df_targets, df_outputs, diseases, path, report_df, chex, testing, single_disease_class=False):
    """
    Here we implement our QC framework. We compare the image labels with the generated report labels.
    As they are modular classifiers this is done one by one.
    """
    print("Quality Control - labels match check.")
    df_targets.to_csv(f"{path}/generated_reports/df_targets.csv")
    #get the image labels for the AC_14. This is from the pretrained image encoder
    ImEncClasses = report_df["ImEncClasses"].apply(lambda x: ast.literal_eval(x))
    ImEncClasses = list(map(list, zip(*[lab for lab in ImEncClasses])))
    ImEncClasses= pd.DataFrame(ImEncClasses, index=diseases).T.astype(int)
    #get image class probabilites
    ImEncProbs = report_df["ClassProb"].apply(lambda x: ast.literal_eval(x))
    ImEncProbs = list(map(list, zip(*[lab for lab in ImEncProbs])))
    ImEncProbs = pd.DataFrame(ImEncProbs, index=diseases).T
    ImEncProbClasses = ImEncProbs[diseases].apply(lambda col: np.where(col >= 0.8,1, np.where(col <= 0.2,0,np.nan)))
    
    if single_disease_class:
        print("sd")
        disease = SD_DISEASE #set in run configs
        #print(report_df["SDImClasses"].head())
        SDImClasses = report_df["SDImClasses"].apply(lambda x: ast.literal_eval(x))
        #print(SDImClasses.head())
        SDImClasses = list(map(list, zip(*[lab for lab in SDImClasses])))
        ImEncClasses= pd.DataFrame(SDImClasses, index=[disease]).T.astype(int)
        
        #get class probabilites
        SDImProbs = report_df["SDImClassProb"].apply(lambda x: ast.literal_eval(x))
        SDImProbs = list(map(list, zip(*[lab for lab in SDImProbs])))
        SDImProbs = pd.DataFrame(SDImProbs, index=[disease]).T
        ImEncProbClasses = SDImProbs[disease].apply(lambda col: np.where(col >= 0.8,1, np.where(col <= 0.2,0,np.nan)))
        ImEncProbClasses = pd.DataFrame(ImEncProbClasses)
        
    
    print("find ids")
    print(df_outputs.head())
    print(report_df.columns)
    print(ImEncClasses.columns)
    print(ImEncProbClasses.columns)
    
    #update take into account uncertainty labels
    accuracy = []
    precision = []
    recall = []
    f1 = []
    bal_accuracy = []
    auc = []
    im_f1 = []
    
    for i, disease in enumerate(diseases):
        if disease == SD_DISEASE: #if AC, commeent out is using all 14 classes.
            print(i, disease)
            print(ImEncProbClasses[disease].value_counts())
            ImEncHiProbClasses_per_disease = ImEncProbClasses[disease].dropna()
            print("ImEncHiProbClasses_per_disease")
            print(f"number of reports: {len(df_outputs)}")
            print(f"Number of cases with {disease} in im enc: {len(ImEncClasses[ImEncClasses[disease] == 1][disease])}")
            print(f"Number of cases with {disease} in output: {len(df_outputs[df_outputs[disease] == 1][disease])}")
            print(f"Number of cases with {disease} in targets: {len(df_targets[df_targets[disease] == 1][disease])}")
            
            #Check for label consistency
            condition1 = ImEncClasses[disease] == df_outputs[disease]
            new_output = df_outputs[condition1][disease] #if just AC or IE
            #if conf threshold is to be applied: 
            new_output = ImEncProbClasses[ImEncProbClasses[disease] == df_outputs[disease]][disease]  #comment out if not using conf threshold
            
            new_targets = df_targets[df_targets.index.isin(new_output.index)][disease]
            kept_ids = report_df[report_df.index.isin(new_output.index)]['ids']
            
            if disease in ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema','Pleural Effusion']:
                if (i ==0) or (single_disease_class==True):
                    overall_matching_ids = kept_ids
                else:
                    overall_matching_ids = overall_matching_ids[overall_matching_ids.isin(kept_ids)]
                print(f"overall_matching_ids: {len(overall_matching_ids)}") #seeing how many for every disease out of curiosity
                
            print(f"number of reports in new output: {len(new_output)}")
            print(f"number of reports in new targets: {len(new_targets)}")
            print(f"number of reports with {disease} in new output: {len(new_output[new_output == 1])}")
            print(f"number of reports with {disease} in new targets: {len(new_targets[new_targets == 1])}")
            if single_disease_class==False:
                kept_ids.to_csv(f"{path}/generated_reports/IE_{disease}_kept_ids.csv") #if IE
            else:
                kept_ids.to_csv(f"{path}/generated_reports/SD_{disease}_kept_ids.csv")
                #kept_ids.to_csv(f"{path}/generated_reports/SD_conf_{disease}_kept_ids.csv")

            if sum(new_output)==0:
                print("No new reports with this disease.")
                accuracy.append(accuracy_score(new_targets, new_output))
                precision.append(np.nan)
                recall.append(np.nan)
                f1.append(np.nan)
                bal_accuracy.append(balanced_accuracy_score(new_targets, new_output))
            else:
                #calc all the clincal metrics and save
                acc = accuracy_score(new_targets, new_output)
                accuracy.append(acc)
                precision.append(precision_score(new_targets, new_output, zero_division=0))
                recall.append(recall_score(new_targets, new_output, zero_division=0))
                f1.append(f1_score(new_targets, new_output, zero_division=0))
                bal_accuracy.append(balanced_accuracy_score(new_targets, new_output))
            print(f"F1: {f1[-1]}")
            print(f"recall: {recall[-1]}")
            print(f"precision: {precision[-1]}")
            print(f"bal acc: {bal_accuracy[-1]}")
            print(f"accuracy: {accuracy[-1]}")
            print(f"Im enc f1 {f1_score(df_targets[disease], ImEncClasses[disease], zero_division=0)}")
            print(f"Im encoder thresholded f1 {f1_score(df_targets[df_targets.index.isin(ImEncHiProbClasses_per_disease.index)][disease], ImEncHiProbClasses_per_disease, zero_division=0)}") 
            #break
    
    if single_disease_class ==False:
        overall_matching_ids.to_csv(f"{path}/generated_reports/overall_five_classes_ids.csv")
    metrics = np.vstack((accuracy, precision, recall, f1, bal_accuracy))
    df_metrics = pd.DataFrame(metrics, columns=diseases, index = ['accuracy','precision','recall','f1','bal_accuracy'])
    #print(df_metrics.round(3))
    #print(df_metrics.dtype())
    mean_ce = df_metrics.mean(axis=1)
    #round the entries in the pandas df to 3 sig figures
    df_metrics = df_metrics.round(3)
    print("Sanity check metrics")
    print(df_metrics)
    print(mean_ce)
    df_metrics.to_csv(f"{path}/generated_reports/{chex}_comp_clinical_metrics.csv")
    mean_ce.to_csv(f"{path}/generated_reports/{chex}_comp_av_clinical_metrics.csv")
    
    return df_metrics

def prepare_labels_for_metrics(chexbert_lab_gen, chexbert_lab_ref):
    #diseases = chexbert_lab_gen.columns[1:]
    diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 
            'Pneumothorax','Support Devices']
    chexbert_lab_gen = labeler14U(diseases, chexbert_lab_gen)
    chexbert_lab_ref = labeler14U(diseases, chexbert_lab_ref)
    #chexbert_lab_gen = labeler14_keepU(diseases, chexbert_lab_gen)
    #chexbert_lab_ref = labeler14_keepU(diseases, chexbert_lab_ref)
    chexbert_lab_gen = chexbert_lab_gen.rename(columns={"Report Impression":"Gen Reports", "labels":"Gen Labels"})
    chexbert_lab_ref = chexbert_lab_ref.rename(columns={"Report Impression":"Ref Reports", "labels":"Ref Labels"})
    testbert_results_df = pd.merge(chexbert_lab_gen, chexbert_lab_ref, left_index=True, right_index=True)
    df_outputs, df_targets = convert_labels(testbert_results_df, diseases) # labels should be the same as targets but are the chexpert ones rather than chexbert
    return df_outputs, df_targets, diseases

def convert_gen_labels(chexbert_lab_gen):
    diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 
            'Pneumothorax','Support Devices']
    chexbert_lab_gen = labeler14U(diseases, chexbert_lab_gen)
    chexbert_lab_gen = chexbert_lab_gen.rename(columns={"Report Impression":"Gen Reports", "labels":"Gen Labels"})
    outputs = list(map(list, zip(*[lab for lab in chexbert_lab_gen["Gen Labels"]]))) 
    df_outputs = pd.DataFrame(outputs, index=diseases).T
    return df_outputs

def convert_gt_labels(rep_df):
    diseases_chexpert = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]
    diseases_mimic = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 
            'Pneumothorax','Support Devices']
    targets = list(map(list, zip(*[ast.literal_eval(lab) for lab in rep_df["disease_labels"]]))) 
    print(len(targets[0]))
    df_targets = pd.DataFrame(targets, index=diseases_mimic).T
    print("df_targets new")
    print(df_targets.shape)
    return df_targets 

#get the nlp metrics. BLUE, cider, meteor, rosie. 
def compute_NLG_scores(gen_sents_or_reports, ref_sents_or_reports):
    def convert_for_pycoco_scorer(sents_or_reports):
        """
        The compute_score methods of the scorer objects require the input not to be list[str],
        but of the form:
        generated_reports =
        {
            "image_id_0" = ["1st generated report"],
            "image_id_1" = ["2nd generated report"],
            ...
        }

        Hence we convert the generated/reference sentences/reports into the appropriate format and also tokenize them
        following Nicolson's (https://arxiv.org/pdf/2201.09405.pdf) and Tanida's (https://arxiv.org/abs/2304.08295) 
        implementation ( (https://github.com/aehrc/cvt2distilgpt2/blob/main/tools/metrics/coco.py):
        """
        sents_or_reports_converted = {}
        for num, text in enumerate(sents_or_reports):
            sents_or_reports_converted[str(num)] = [re.sub(' +', ' ', text.replace(".", " ."))]

        return sents_or_reports_converted
    """
    Computes NLG metrics that are specified in metrics list (1st input argument):
        - Bleu 1-4
        - Meteor
        - Rouge-L
        - Cider-D <- this one doesnt work for some reason - come back to.

    Returns a dict that maps from the metrics specified to the corresponding scores.
    """
    nlg_metrics = ["bleu", "meteor", "rouge"]  # "cider" is currently not supported
    scorers = {}
    if "bleu" in nlg_metrics:
        scorers["bleu"] = Bleu(4)
    if "meteor" in nlg_metrics:
        scorers["meteor"] = Meteor()
    if "rouge" in nlg_metrics:
        scorers["rouge"] = Rouge()  # this is actually the Rouge-L score, even if the class name only says Rouge
    if "cider" in nlg_metrics:
        scorers["cider"] = Cider()  # this is actually the Cider-D score, even if the class name only says Cider

    gen_sents_or_reports = convert_for_pycoco_scorer(gen_sents_or_reports)
    ref_sents_or_reports = convert_for_pycoco_scorer(ref_sents_or_reports)

    nlg_scores = {}

    for metric_name, scorer in scorers.items():
        score, _ = scorer.compute_score(ref_sents_or_reports, gen_sents_or_reports)
        if metric_name == "bleu":
            nlg_scores["bleu_1"] = score[0]
            nlg_scores["bleu_2"] = score[1]
            nlg_scores["bleu_3"] = score[2]
            nlg_scores["bleu_4"] = score[3]
        else:
            nlg_scores[metric_name] = score

    nlp_metrics_df = pd.DataFrame(list(nlg_scores.values()), columns=['metrics'], index=nlg_scores.keys())
    return nlp_metrics_df

def get_all_metrics(report_df, FULL_RUN, path, epoch, single_disease_class,  testing=False, val = False):
    chexbert = get_chexbert()
    #Get the report labels using chexbert labeler
    preds_gen_reports, preds_ref_reports, gen_df, ref_df = get_chexbert_labels_for_gen_and_ref_reports(chexbert, report_df, FULL_RUN, epoch, testing=testing)
    
    df_outputs, df_targets, diseases = prepare_labels_for_metrics(gen_df, ref_df) #when working from chexbert labels extracted from ref report findings.
    #when working from reference labels
    #df_outputs = convert_gen_labels(gen_df)
    #df_targets = convert_gt_labels(report_df) #change disease order depeding on which test set.(bro thats inconvinient change at some point)
    
    #df_targets = df_targets[df_outputs.columns] #make sure the columns are the same
    clinical_metrics, mean_ce = ce_metrics(df_targets, df_outputs, diseases, path, chex="chexbert", testing=testing, val=val)
    
    #If we have evaluated the test set and we want to apply tthe QC framework as well.
    if (val == False): 
        ce_matching = ce_metrics_disease_match(df_targets, df_outputs, diseases, path, report_df,chex="chexbert", testing=testing, single_disease_class=single_disease_class)

    #chexpert
    nlp_metrics_df = compute_NLG_scores(report_df["gen_reports"], report_df["ref_reports"]) 
    #stick nlp and ce metrics together
    all_metrics_df = pd.concat([mean_ce, nlp_metrics_df])
    all_metrics_df.to_csv(f"{path}/generated_reports/all_metrics.csv")
    return all_metrics_df

def test_main(FULL_RUN, no_im_tokens, epoch =None,im_mod_retrain=False, testing=False, single_disease_class=False):
    ### get test generated reports ###
    if CONTINUE_TRAINING:
        print("testing on continued training model.")
        path = f"/src/runs/full_model/run_{FULL_RUN}/cont/generated_sentences_and_reports"
    else:
        if epoch == None:
            path = f"/src/runs/full_model/run_{FULL_RUN}/generated_sentences_and_reports"
        else:
            path = f"/src/runs/full_model/run_{FULL_RUN}/generated_sentences_and_reports_e{epoch}"

    start_time = time.time()
    tokeniser = get_tokenizer()
    
    model, test_dl, ImageEncoder = setup(FULL_RUN, tokeniser, epoch, no_im_tokens =no_im_tokens, im_mod_retrain=im_mod_retrain, testing=testing)
    report_df  = test(test_dl, model, tokeniser, ImageEncoder, im_mod_retrain, FULL_RUN, path, no_im_tokens=no_im_tokens, testing=testing, single_disease_class = single_disease_class)
    # if already run previous
    #report_df = pd.read_csv(f"/src/runs/full_model/run_{FULL_RUN}/generated_sentences_and_reports_e{epoch}/test_ImClasses.csv") #test_reports.csv 
    
    ### Evaluate model ###
    all_metrics_df = get_all_metrics(report_df, FULL_RUN, path, epoch, single_disease_class, testing)
    print(all_metrics_df)
    end_time = time.time()
    elapsed_time = end_time - start_time
    log.info(f'testing time: {elapsed_time/(60*60)} hrs')

    print("Finished testing.")

if __name__ == "__main__":
    FULL_RUN = 36
    print("test run: ", FULL_RUN)
    no_im_tokens = NO_IMAGE_TOKENS
    im_mod_retrain = True # False
    single_disease_class = False
    testing= False  
    epochs = [10,20,30,40]
    epochs = [20]
    for epoch in epochs:
        print(f"testing epoch {epoch}")
        test_main(FULL_RUN, no_im_tokens, epoch, im_mod_retrain, testing, single_disease_class =single_disease_class)