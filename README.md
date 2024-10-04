# GenX: Auditing Generated Radiology Reports 

Code for the paper [Quality Control for Radiology Report Generation
Models via Auxiliary Auditing Components](https://arxiv.org/pdf/2407.21638)

Accepted to the UNSURE workshop at MICCAI 2024.

## Abstract

Automation of medical image interpretation could alleviate bottlenecks in diagnostic workflows, and has become of particular interest in recent years due to advancements in natural language processing. Great strides have been made towards automated radiology report generation via AI, yet ensuring clinical accuracy in generated reports is a significant challenge, hindering deployment of such methods in clinical practice. In this work we propose a quality control framework for assessing the reliability of AI-generated radiology reports with respect to semantics of diagnostic importance using modular auxiliary auditing components (ACs). Evaluating our pipeline on the MIMIC CXR dataset, our findings show that incorporating ACs in the form of disease-classifiers can enable auditing that identifies more reliable reports, resulting in higher F1 scores compared to unfiltered generated reports. Additionally, leveraging the confidence of the AC labels further improves the audit’s effectiveness.

## Method

<img src="https://github.com/user-attachments/assets/55414d45-723c-40dc-9ea8-b741a86ad7a8" width="900">

*Fig. 1: Proposed error detection pipeline of radiology report generation using auxiliary auditing components. The standard report generation pipeline (a) is followed by the CheXbert labeler, $g_T$ (b) that extracts pathology labels, $C_T$, from the reports, that are semantically meaningful for clinical diagnosis. (c) Modular image-based audit models, AC, here disease classifiers, predict disease class labels, $C_I$ , based on image, $I$. (d) If the labels predicted based on image ($C_I$) and report ($C_T$) are consistent, the report’s contents are deemed likely reliable. In case of inconsistency, $C_I \neq C_T$ , the report is flagged as less reliable, potentially containing an error. If the image-based classifiers have predictive confidence below threshold $t$, auditing can be deferred to user due to uncertainty.*

## Results

<img src="https://github.com/user-attachments/assets/283db9f8-b895-48ff-9bef-66b74fa3c705" width="900">

*Table 3: Evaluation on test split of our auditing framework. For the 5 most commonly reported classes in literature, GenX column reports F1 score of all generated reports ($C_T$ vs true label). $AC_1$ assesses image-based classification by 5 disease-specific ACs ($C_I$ vs true label). $GenX +AC^{t=0}_{1}$ and $GenX +AC^{t=0.8}_{1}$ show semantic factuality of generated reports that passed auditing for the specific disease against $AC_1$, without the requirement for confidence $p_{AC} ≥ t$ and when confidence over t = 0.8 is required, respectively. Bold shows improvement by auditing over baseline GenX. Parentheses show percentage of reports that satisfy the auditing per disease. We also report results when auditing with a single multi-label AC trained for all 14 classes ($AC_{14}$ and $GenX+AC^{t=0}_{14}$ ). Developing independent ACs per disease is easier and more effective in practice*

## Running code

To install the required packages, run:

```
pip install -r requirements.txt
```

Download the mimic data here: https://physionet.org/content/mimic-cxr/2.1.0/  
and the jpgs and labels here: https://physionet.org/content/mimic-cxr-jpg/2.1.0/

Data preprocessing details can be found here: [README.md](https://github.com/hermionewarr/GenX_Report_Audit/blob/main/src/dataset/README.md).

For the different models adjust the run configurations in **src/utils/run_configurations.py**. Specify data locations in **src/utils/path_datasets_and_weights.py**.

To pretrain the image encoder model, navigate to the **/src/vision_model folder** and run the **train_image_model.py** script.

To train the report generation model, GenX, go to **/src/full_model/train_full_model.py**. 

To evaluate the model and use the qualtity control framework in the same full_model folder run the **test_and_QC.py** script.

## Citation

When citing this work please use:

```
 @inproceedings{Warr_2025, 
   title={Quality Control for Radiology Report Generation Models via Auxiliary Auditing Components}, 
   DOI={10.1007/978-3-031-73158-7_7}, 
   booktitle={Uncertainty for Safe Utilization of Machine Learning in Medical Imaging}, 
   publisher={Springer Nature Switzerland}, 
   author={Warr, Hermione and Ibrahim, Yasin and McGowan, Daniel R. and Kamnitsas, Konstantinos}, 
   editor={Sudre, Carole H. and Mehta, Raghav and Ouyang, Cheng and Qin, Chen and Rakic, Marianne and Wells, William M.}, 
   year={2024}, 
   pages={70–80}
 }
```

## Acknowledgements

Code for the report generator GenX was adapted from:

- https://github.com/ttanida/rgrg  
  Tanida, T., et al.: Interactive and explainable region-guided radiology report generation. In: CVPR (2023) 

