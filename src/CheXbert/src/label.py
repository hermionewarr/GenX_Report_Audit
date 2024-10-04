import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import OrderedDict

#from CheXbert.src import utils
import utils
from CheXbert.src.models.bert_labeler import bert_labeler
#from models.bert_labeler import bert_labeler
from CheXbert.src.datasets_chexbert.unlabeled_dataset import UnlabeledDataset
#from datasets_chexbert.unlabeled_dataset import UnlabeledDataset
from CheXbert.src.constants import *
#from constants import *
from CheXbert.src.utils import generate_attention_masks
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn_no_labels(sample_list):
    """Custom collate function to pad reports in each batch to the max len,
       where the reports have no associated labels
    @param sample_list (List): A list of samples. Each sample is a dictionary with
                               keys 'imp', 'len' as returned by the __getitem__
                               function of ImpressionsDataset

    @returns batch (dictionary): A dictionary with keys 'imp' and 'len' but now
                                 'imp' is a tensor with padding and batch size as the
                                 first dimension. 'len' is a list of the length of 
                                 each sequence in batch
    """
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(tensor_list,
                                                  batch_first=True,
                                                  padding_value=PAD_IDX)
    len_list = [s['len'] for s in sample_list]
    batch = {'imp': batched_imp, 'len': len_list}
    return batch

def load_unlabeled_data(csv_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        shuffle=False):
    """ Create UnlabeledDataset object for the input reports
    @param csv_path (string): path to csv file containing reports
    @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                             that can fit on a TITAN XP is 6 if the max sequence length
                             is 512, which is our case. We have 3 TITAN XP's
    @param num_workers (int): how many worker processes to use to load data
    @param shuffle (bool): whether to shuffle the data or not  
    
    @returns loader (dataloader): dataloader object for the reports
    """
    collate_fn = collate_fn_no_labels
    dset = UnlabeledDataset(csv_path)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, collate_fn=collate_fn)
    return loader
    
def label(checkpoint_path, csv_path):
    """Labels a dataset of reports
    @param model (nn.Module): instantiated CheXbert model
    @param csv_path (string): location of csv with reports

    @returns y_pred (List[List[int]]): Labels for each of the 14 conditions, per report  
    """
    ld = load_unlabeled_data(csv_path)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #for running from cli
    ####
    model = bert_labeler()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    ####
    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]

    print("\nBegin report impression labeling. The progress bar counts the # of batches completed:")
    print("The batch size is %d" % BATCH_SIZE)
    print("Chexpert: ", device)
    with torch.no_grad():
        for i, data in enumerate(tqdm(ld)):
            batch = data['imp']  # (batch_size, max_len)
            batch = batch.to(device)
            src_len = data['len']
            attn_mask = generate_attention_masks(batch, src_len, device)
            
            out = model(batch, attn_mask)
            #print(out.size())
            for j in range(len(out)):
                curr_y_pred = out[j].argmax(dim=1)  # shape is (batch_size)
                y_pred[j].append(curr_y_pred)

        for j in range(len(y_pred)):
            y_pred[j] = torch.cat(y_pred[j], dim=0)

    y_pred = [t.tolist() for t in y_pred]
    return y_pred

def save_preds(y_pred, csv_path, out_path, extra_name):
    """Save predictions as out_path/labeled_reports.csv 
    @param y_pred (List[List[int]]): list of predictions for each report
    @param csv_path (string): path to csv containing reports
    @param out_path (string): path to output directory
    """
    y_pred = np.array(y_pred)
    y_pred = y_pred.T
    print("y_pred: ", y_pred.shape)
    
    df = pd.DataFrame(y_pred, columns=CONDITIONS)
    reports = pd.read_csv(csv_path)['Report Impression']

    df['Report Impression'] = reports.tolist()
    new_cols = ['Report Impression'] + CONDITIONS
    df = df[new_cols]

    df.replace(0, np.nan, inplace=True) #blank class is NaN
    df.replace(3, -1, inplace=True)     #uncertain class is -1
    df.replace(2, 0, inplace=True)      #negative class is 0 
    
    df.to_csv(os.path.join(out_path, f'labeled_reports_{extra_name}.csv'), index=False)
    #df.to_csv(out_path, index=False)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label a csv file containing radiology reports')
    parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                        help='path to csv containing reports. The reports should be \
                              under the \"Report Impression\" column')
    parser.add_argument('-o', '--output_dir', type=str, nargs='?', required=True,
                        help='path to intended output folder')
    parser.add_argument('-c', '--checkpoint', type=str, nargs='?', required=True,
                        help='path to the pytorch checkpoint')
    args = parser.parse_args()
    csv_path = args.data
    out_path = args.output_dir
    checkpoint_path = args.checkpoint

    y_pred = label(checkpoint_path, csv_path)
    save_preds(y_pred, csv_path, out_path)
