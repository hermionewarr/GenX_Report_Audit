"""
Script creates train.csv, valid.csv, and test.csv to train image encoder (as a standalone module),
and full model.

Have to change whether or not to include patients without findings ref report. im encoder report findings doenst matter. full model it does.

Each row in the csv files specifies information about a single image.

The specific information (i.e. columns) of each row are:
    - subject_id (str): id of the patient whose image is used
    - study_id (str): id of the study of that patient (since a patient can have several studies done to document the progression of a disease etc.)
    - image_id (str): id of the single image
    - mimic_image_file_path (str): file path to the jpg of the single image 
    - reference_report (str): the "findings" section of the MIMIC-CXR report corresponding to the image (see function get_reference_report)
    - or report_impressions if decide to use those
    - labels
    - weight/freq <- need to add if want to oversample

Once data created in initial_data folder then can move to full_model_data_path or im_model_datat_path depending on which youve created.
"""

import csv
import json
import logging
import os
import torch
from tqdm import tqdm
import re
import pandas as pd

import section_parser as sp
from constants import IMAGE_IDS_TO_IGNORE, SUBSTRINGS_TO_REMOVE
import sys
sys.path.insert(0,"/home/hermione/Documents/VLP/GenX_paper_clean/")
print(sys.path)
from src.utils.path_datasets_and_weights import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# to log certain statistics during dataset creation
txt_file_for_logging = "log_file_dataset_creation.txt"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# constant specifies how many rows to create in the customized csv files
# can be useful to create small sample datasets (e.g. of len 200) for testing things
# if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES is None, then all possible rows are created
NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES = 10 #None


def write_stats_to_log_file(
    dataset: str,
    num_images_ignored_or_avoided: int,
    missing_images: list[str],
    missing_reports: list[str],
):
    with open(txt_file_for_logging, "a") as f:
        f.write(f"{dataset}:\n")
        f.write(f"\tnum_images_ignored_or_avoided: {num_images_ignored_or_avoided}\n")

        f.write(f"\tnum_missing_images: {len(missing_images)}\n")
        for missing_img in missing_images:
            f.write(f"\t\tmissing_img: {missing_img}\n")

        f.write(f"\tnum_missing_reports: {len(missing_reports)}\n")
        for missing_rep in missing_reports:
            f.write(f"\t\tmissing_rep: {missing_rep}\n")


def write_rows_in_new_csv_file(dataset: str, csv_rows: list[list]) -> None:
    log.info(f"Writing rows into new {dataset}.csv file...")

    new_csv_file_path = os.path.join(path_full_dataset, dataset)
    new_csv_file_path += ".csv" if not NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES else f"-{NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES}.csv"

    #header = ["subject_id", "study_id", "image_id", "mimic_image_file_path", "reference_report"]
    header = ["subject_id", "study_id", "image_id", "mimic_image_file_path", "reference_report", "findings", "impression", "indication", "history"]


    with open(new_csv_file_path, "w") as fp:
        csv_writer = csv.writer(fp)

        csv_writer.writerow(header)
        csv_writer.writerows(csv_rows)
    return new_csv_file_path

def convert_phrases_to_single_string(phrases: list[str], sentence_tokenizer) -> str:
    """
    Takes a list of phrases describing the region of a single bbox and returns a single string.

    Also performs operations to clean the single string, such as:
        - removes irrelevant substrings (like "PORTABLE UPRIGHT AP VIEW OF THE CHEST:")
        - removes whitespace characters (e.g. \n or \t) and redundant whitespaces
        - capitalizes the first word in each sentence
        - removes duplicate sentences

    Args:
        phrases (list[str]): in the attribute dictionary, phrases is originally a list of strings
        sentence_tokenizer (spacy sentence tokenizer): used in capitalize_first_word_in_sentence function

    Returns:
        phrases (str): a single string, with the list of strings concatenated
    """
    def remove_substrings(phrases):
        def remove_wet_read(phrases):
            """Removes substring like 'WET READ: ___ ___ 8:19 AM' that is irrelevant."""
            # since there can be multiple WET READS's, collect the indices where they start and end in index_slices_to_remove
            index_slices_to_remove = []
            for index in range(len(phrases)):
                if phrases[index:index + 8] == "WET READ":

                    # curr_index searches for "AM" or "PM" that signals the end of the WET READ substring
                    for curr_index in range(index + 8, len(phrases)):
                        # since it's possible that a WET READ substring does not have an"AM" or "PM" that signals its end, we also have to break out of the iteration
                        # if the next WET READ substring is encountered
                        if phrases[curr_index:curr_index + 2] in ["AM", "PM"] or phrases[curr_index:curr_index + 8] == "WET READ":
                            break

                    # only add (index, curr_index + 2) (i.e. the indices of the found WET READ substring) to index_slices_to_remove if an "AM" or "PM" were found
                    if phrases[curr_index:curr_index + 2] in ["AM", "PM"]:
                        index_slices_to_remove.append((index, curr_index + 2))

            # remove the slices in reversed order, such that the correct index order is preserved
            for indices_tuple in reversed(index_slices_to_remove):
                start_index, end_index = indices_tuple
                phrases = phrases[:start_index] + phrases[end_index:]

            return phrases

        phrases = remove_wet_read(phrases)
        phrases = re.sub(SUBSTRINGS_TO_REMOVE, "", phrases, flags=re.DOTALL)

        return phrases

    def remove_whitespace(phrases):
        phrases = " ".join(phrases.split())
        return phrases

    def capitalize_first_word_in_sentence(phrases, sentence_tokenizer):
        sentences = sentence_tokenizer(phrases).sents

        # capitalize the first letter of each sentence
        phrases = " ".join(sent.text[0].upper() + sent.text[1:] for sent in sentences)

        return phrases

    def remove_duplicate_sentences(phrases):
        # remove the last period
        if phrases[-1] == ".":
            phrases = phrases[:-1]

        # dicts are insertion ordered as of Python 3.6
        phrases_dict = {phrase: None for phrase in phrases.split(". ")}

        phrases = ". ".join(phrase for phrase in phrases_dict)

        # add last period
        return phrases + "."

    # convert list of phrases into a single phrase
    phrases = " ".join(phrases)

    # remove "PORTABLE UPRIGHT AP VIEW OF THE CHEST:" and similar substrings from phrases, since they don't add any relevant information
    phrases = remove_substrings(phrases)

    # remove all whitespace characters (multiple whitespaces, newlines, tabs etc.)
    phrases = remove_whitespace(phrases)

    # for consistency, capitalize the 1st word in each sentence
    phrases = capitalize_first_word_in_sentence(phrases, sentence_tokenizer)

    phrases = remove_duplicate_sentences(phrases)

    return phrases


def get_reference_report(subject_id: str, study_id: str, missing_reports: list[str]):
    # custom_section_names and custom_indices specify reports that don't have "findings" sections
    custom_section_names, custom_indices = sp.custom_mimic_cxr_rules()

    if f"s{study_id}" in custom_section_names or f"s{study_id}" in custom_indices:
        return -1  # skip all reports without "findings" sections

    path_to_report = os.path.join(path_mimic_cxr, "files", f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id}.txt")

    if not os.path.exists(path_to_report):
        shortened_path_to_report = os.path.join(f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id}.txt")
        missing_reports.append(shortened_path_to_report)
        return -1

    with open(path_to_report) as f:
        report = "".join(f.readlines())

    # split report into sections
    # section_names is a list that specifies the found sections, e.g. ["indication", "comparison", "findings", "impression"]
    # sections is a list of same length that contains the corresponding text from the sections specified in section_names
    sections, section_names, _ = sp.section_text(report)

    if "findings" in section_names:
        # get index of "findings" by matching from reverse (has to do with how section_names is constructed)
        findings_index = len(section_names) - section_names[-1::-1].index("findings") - 1
        report = sections[findings_index]
    else:
        return -1  # skip all reports without "findings" sections

    # remove unnecessary whitespaces
    report = " ".join(report.split())
    #print("report found")
    return report

def get_reference_report_all_text(subject_id: str, study_id: str, missing_reports: list[str]):
    # custom_section_names and custom_indices specify reports that don't have "findings" sections
    custom_section_names, custom_indices = sp.custom_mimic_cxr_rules()
    report=0
    if f"s{study_id}" in custom_section_names or f"s{study_id}" in custom_indices:
        #return -1  # skip all reports without "findings" sections changed this as we dont need reports for image encoder pretraininig
        report=-1

    path_to_report = os.path.join(path_mimic_cxr, "files", f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id}.txt")

    if not os.path.exists(path_to_report):
        shortened_path_to_report = os.path.join(f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id}.txt")
        missing_reports.append(shortened_path_to_report)
        #return -1
        report = -1
    
    sections_list = ["indication", "history", "findings", "impression"]
    section_values = {section: None for section in sections_list}

    if report !=-1:
        with open(path_to_report) as f:
            report = "".join(f.readlines())

        # split report into sections
        # section_names is a list that specifies the found sections, e.g. ["indication", "comparison", "findings", "impression"]
        # sections is a list of same length that contains the corresponding text from the sections specified in section_names
        sections, section_names, _ = sp.section_text(report)

        for report_section in sections_list:
            if report_section in section_names:
                index = len(section_names) - section_names[-1::-1].index(report_section) - 1
                section_values[report_section] = (' '.join(sections[index].split())).replace('_','').replace(' //','.').lstrip(' ')
            else:
                section_values[report_section] = None 

        report = '\n'.join([f"{section}: {section_values[section]}" for section in sections_list if section_values[section] is not None])

    
    # remove unnecessary whitespaces
        report = " ".join(report.split())
    
    return report, section_values["findings"], section_values["impression"], section_values["indication"], section_values["history"]


def get_total_num_rows(path_csv_file: str) -> int:
    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        return sum(1 for row in csv_reader)


def get_rows(dataset: str, path_csv_file: str) -> list[list]:
    """
    Args:
        dataset (str): either "train", "valid" or "test
        path_csv_file (str): path to one of the csv files in the folder silver_dataset/splits of the chest-imagenome-dataset
        image_ids_to_avoid (set): as specified in "silver_dataset/splits/images_to_avoid.csv"

    Returns:
        csv_rows (list[list]): inner list contains information about a single image:
            - subject_id (str)
            - study_id (str)
            - image_id (str)
            - mimic_image_file_path (str): file path to image in mimic-cxr-jpg dataset
            - reference_report (str): the findings section of the report extracted via https://github.com/MIT-LCP/mimic-cxr/tree/master/txt
    """
    csv_rows = []
    num_rows_created = 0

    total_num_rows = get_total_num_rows(path_csv_file)
    print(total_num_rows)

    # stats will be logged in path_to_log_file
    num_images_ignored_or_avoided = 0
    missing_images = []
    missing_reports = []

    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        # iterate over all rows of the given csv file (i.e. over all images), if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES is not set to a specific value
        for no, row in tqdm(enumerate(csv_reader), total=total_num_rows):
            subject_id = row[3]
            study_id = row[2]
            image_id = row[1]
            # all images in set IMAGE_IDS_TO_IGNORE seem to be failed x-rays and thus have to be discarded
            if image_id in IMAGE_IDS_TO_IGNORE: 
                num_images_ignored_or_avoided += 1
                continue
            
            """ if no >= NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES:
                print(f"reached {NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES} images.")
                break """
            # image_file_path is of the form "files/p10/p10000980/s50985099/6ad03ed1-97ee17ee-9cf8b320-f7011003-cd93b42d.dcm"
            # i.e. f"files/p{subject_id[:2]}/p{subject_id}/s{study_id}/{image_id}.dcm"
            # since we have the MIMIC-CXR-JPG dataset, we need to replace .dcm by .jpg
            image_file_path = row[4]
            mimic_image_file_path = os.path.join(path_mimic_cxr_jpg, image_file_path)
           
            if not os.path.exists(mimic_image_file_path):
                missing_images.append(mimic_image_file_path)
                continue
         
            # for the validation and test sets, we only want to include images that have corresponding reference reports with "findings" sections
            reference_report, findings, impressions, indication, history = get_reference_report_all_text(subject_id, study_id, missing_reports)
          
            # skip images that don't have a reference report with "findings" section - only for language model.
            # otherwise, we would have to skip all images that don't have a reference report.
            if reference_report == -1:
                continue

            # new_image_row will store all information about 1 image as a row in the csv file
            new_image_row = [subject_id, study_id, image_id, mimic_image_file_path,reference_report, findings, impressions, indication, history]
           
            csv_rows.append(new_image_row)
            num_rows_created += 1
            # break out of loop if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES is specified
            if num_rows_created == NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES:
                print(f"reached {NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES} images.")
                break
            
    write_stats_to_log_file(dataset, num_images_ignored_or_avoided, missing_images, missing_reports)
    print(num_rows_created)
    return csv_rows


def create_new_csv_file(dataset: str, path_csv_file: str) -> None:
    log.info(f"Creating new {dataset}.csv file...")

    # get rows to create new csv_file
    # csv_rows is a list of lists, where an inner list specifies all information about a single image
    csv_rows = get_rows(dataset, path_csv_file)

    # write those rows into a new csv file
    new_csv_file_path = write_rows_in_new_csv_file(dataset, csv_rows)

    log.info(f"Creating new {dataset}.csv file... DONE!")
    return new_csv_file_path


def create_new_csv_files(csv_files_dict):
    if os.path.exists(path_full_dataset):
        log.error(f"Full dataset folder already exists at {path_full_dataset}.")
        log.error("Delete dataset folder or rename variable path_full_dataset in src/path_datasets_and_weights.py before running script to create new folder!")
        return None

    os.mkdir(path_full_dataset)
    csv_names = []
    for dataset, path_csv_file in csv_files_dict.items():
        new_csv_file_path = create_new_csv_file(dataset, path_csv_file)
        csv_names.append(new_csv_file_path)
    return csv_names


def get_train_val_test_csv_files():
    """Return a dict with datasets as keys and paths to the corresponding csv files in the chest-imagenome dataset as values"""
    #path_to_splits_folder = os.path.join(path_chest_imagenome, "silver_dataset", "splits")
    path_to_splits_folder = path_mimic_split
    return {dataset: os.path.join(path_to_splits_folder, dataset) + ".csv" for dataset in ["train", "valid", "test"]}

#add the disease labels to the files
def labeler(classes,diseases, df, U=True, test=False):
    # Add a new column ‘labels’ to the DataFrame and assign lists of 0s and 1s based on diseases
    # Default treats uncertain labels as positive.
    print("df len: ", len(df))
    #check that cant be no findings with any positive disease mention - basic sanity check
    df = df[~((df['No Finding'] == 1.0) & (df[diseases].eq(1).any(axis=1)))]
    print("df len sanity checked: ",len(df)) #see if you lose any.
    #disease labeler
    if test:
        #keep the uncertainty labels
        print("U=-1")
        df['labels'] = df[classes].apply(lambda row: [1 if (row[disease] == 1) else -1 if (row[disease] ==-1) else 0 for disease in classes], axis=1)
    elif U==True:
        #U=+ve - make uncertain labels positive. the case we end up using.
        print("U=+ve")
        df['labels'] = df[classes].apply(lambda row: [1 if (row[disease] == 1 or row[disease] ==-1) else 0 for disease in classes], axis=1)
    else:
        #U=-ve  - investigate the case where the uncertain labels are treated as negative.
        print("U=-ve")
        df['labels'] = df[classes].apply(lambda row: [1 if (row[disease] == 1) else 0 for disease in classes], axis=1)
    df = df.drop(columns=classes)
    #df = df.drop(columns=['reference_report'])
    return df

def combine_mimic_metadata(csv_names):
    #print(csv_names)
    #load in files weve just created plus the chexpert mimic labels
    chexpert = pd.read_csv(path_mimic_chexpert_labels)
    train = pd.read_csv(f"{path_full_dataset}/{csv_names[0].split('/')[-1]}")
    val = pd.read_csv(f"{path_full_dataset}/{csv_names[1].split('/')[-1]}")
    test = pd.read_csv(f"{path_full_dataset}/{csv_names[2].split('/')[-1]}")
    train_chex = pd.merge(train, chexpert, on=['subject_id', 'study_id'])
    val_chex = pd.merge(val, chexpert, on=['subject_id', 'study_id'])
    test_chex = pd.merge(test, chexpert, on=['subject_id', 'study_id'])
    classes = chexpert.columns[2:]
    diseases = classes.drop('No Finding')
    #label the images
    train_chex = labeler(classes,diseases, train_chex)
    val_chex = labeler(classes,diseases, val_chex)
    test_chex = labeler(classes,diseases, test_chex)
    #if have used downsized images then need to change the file paths - i added _small to the file names so need to insert that.
    train_chex['mimic_image_file_path'] = train_chex['mimic_image_file_path'].str.replace('.jpg','_small.jpg')
    val_chex['mimic_image_file_path'] = val_chex['mimic_image_file_path'].str.replace('.jpg','_small.jpg')
    test_chex['mimic_image_file_path'] = test_chex['mimic_image_file_path'].str.replace('.jpg','_small.jpg')
    #for the image encoder we also tried various oversampling methods and had an additional csv column of the weights to supply to the dataloader.
    #can be added here.
    #save the files
    train_chex.to_csv(f"{path_full_dataset}/train_disease_labels.csv", index=False)
    val_chex.to_csv(f"{path_full_dataset}/valid_disease_labels.csv", index=False)
    test_chex.to_csv(f"{path_full_dataset}/test_disease_labels.csv", index=False)
    return

def main():
    csv_files_dict = get_train_val_test_csv_files()
    #create initial data that contains mimic reports for each image as a csv 
    csv_names = create_new_csv_files(csv_files_dict)
    #label each image with the provided chexpert labels
    combine_mimic_metadata(csv_names)

if __name__ == "__main__":
    main()
