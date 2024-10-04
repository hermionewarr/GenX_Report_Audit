"""
MIMIC-CXR and MIMIC-CXR-JPG dataset paths should both have a (sub-)directory called "files" in their directories.

path_chexbert_weights specifies the path to the weights of the CheXbert labeler needed to extract the disease labels from the generated and reference reports.
The weights can be downloaded here: https://github.com/stanfordmlgroup/CheXbert#checkpoint-download

path_runs_* specify the directories where all the run folders (containing checkpoints, etc.) will be created
when training the image encoder and full model (with and without the language model).
That means the directories specified by path_runs_* should already exist before starting the training.
"""

DATA_PATH = "path/GenX_Report_Audit/example_data/"
MODEL_INFO_PATH = "path/GenX_Report_Audit/"
path_mimic_cxr = "path/physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports/"
path_mimic_split = "path/dataset-with-reference-reports/splits_mimic/"
path_mimic_cxr_jpg = "path/physionet.org/files/mimic-cxr-jpg/2.0.0/"
path_mimic_chexpert_labels = "path/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv"
path_full_dataset= DATA_PATH + "initial_data"
path_full_dataset_im = DATA_PATH + "im_model_data" #just im model
path_full_dataset_full = DATA_PATH + "full_model_data" #full model if its some where different
#chexbert labeller weights
path_chexbert_weights = "/GenX_Report/src/CheXbert/src/models/chexbert.pth" #you will need to download this
#where you want to save the model
path_runs_image_model = MODEL_INFO_PATH+"src/runs/image_model"
path_runs_full_model = MODEL_INFO_PATH+"src/runs/full_model"
