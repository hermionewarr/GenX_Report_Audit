import torch
from utils.run_configurations import PRETRAIN_CNN, PRETRAIN_LLM, TRAIN_FULL_MODEL

class CustomCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str]]):
        """
        batch is a list of dicts where each dict corresponds to a single image and has the keys:
          - image
          - labels
          - input_ids
          - attention_mask
          - reference_report
        """
        # discard samples from batch where __getitem__ from custom_dataset failed (i.e. returned None)
        # otherwise, whole training loop would stop
        batch = list(filter(lambda x: x is not None, batch))  # filter out samples that are None

        # allocate an empty tensor images_batch that will store all images of the batch
        image_size = batch[0]["image"].size()
        images_batch = torch.empty(size=(len(batch), *image_size))

        # create an empty list image_targets that will store dicts containing the bbox_coordinates and bbox_labels
        #image_targets = []
        reference_reports = []
        labels = []
        ids = []

        for i, sample_dict in enumerate(batch):
            # remove image tensors from batch and store them in dedicated images_batch tensor
            #print(sample_dict.keys())
            images_batch[i] = sample_dict.pop("image")
            labels.append(sample_dict.pop("labels"))
            if TRAIN_FULL_MODEL:
                reference_reports.append(sample_dict.pop("reference_report"))
                ids.append(sample_dict.pop("mimic_image_file_path"))

        if PRETRAIN_CNN:
            batch = {}
        else:
            # batch is now a list that only contains dicts with keys input_ids and attention_mask (both of which are List[List[int]])
            # i.e. batch is of type List[Dict[str, List[List[int]]]]
            # each dict specifies the input_ids and attention_mask of a single image, 
            # we want to pad all input_ids and attention_mask to the max sequence length in the batch
            # we can use the pad method of the tokenizer for this, however it requires the input to be of type Dict[str, List[List[int]]
            # thus we first transform the batch into a dict with keys "input_ids" and "attention_mask", both of which are List[List[int]]
            # that hold the input_ids and attention_mask of all the regions in the batch (i.e. the outer list will have (batch_size) elements)
            
            dict_with_ii_and_am = self.transform_to_dict_with_inputs_ids_and_attention_masks(batch)

            # we can now apply the pad method, which will pad the input_ids and attention_mask to the longest sequence in the batch
            # the keys "input_ids" and "attention_mask" in dict_with_ii_and_am will each map to a tensor of shape [(batch_size), (longest) seq_len (in batch)]
            dict_with_ii_and_am = self.tokenizer.pad(dict_with_ii_and_am, padding="longest", return_tensors="pt", max_length=512) #padding="longest",padding='max_length'
            #print(dict_with_ii_and_am)
            # treat dict_with_ii_and_am as the batch variable now (since it is a dict, and we can use it to store all the other keys as well)
            batch = dict_with_ii_and_am
            batch["reference_reports"] = reference_reports
            batch["mimic_image_file_path"] = ids

        # add the remaining keys and values to the batch dict
        batch["image"] = images_batch
        batch["labels"] = labels
        

        return batch

    def transform_to_dict_with_inputs_ids_and_attention_masks(self, batch):
        dict_with_ii_and_am = {"input_ids": [], "attention_mask": []}
        for single_dict in batch:
            for key, outer_list in single_dict.items():
                for inner_list in outer_list:
                    dict_with_ii_and_am[key].append(inner_list)

        return dict_with_ii_and_am
