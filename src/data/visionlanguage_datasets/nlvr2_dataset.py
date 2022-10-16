import sys
import os
import time
import json
import jsonlines
import logging
import glob
from tqdm import tqdm
import pickle
import pdb
from PIL import Image
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
import random

from PIL import Image
from torchvision import transforms as T

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)



class NLVR2Dataset(Dataset):

    def __init__(self, 
                 data_dir: str, 
                 split: str,
                 **kwargs):

        """
        Initiates the NLVR2Dataset - loads all the sentences and corresponding image IDs and output label 
        Every item in self.data corresponds to a single NLVR2 input

        Args:
        data_dir : path containing NLVR2 annotations and images
        split: either train/val/test split

        Returns:
        Loads all annotations into self.data, where each item is a single NLVR2 input
        """

        self.data_dir = data_dir
        self.num_labels = 2
        self.split = split

        rename_split = {'train': 'train', 'val': 'dev', 'test': 'test1'}
        _split = rename_split[split]
        self.image_dir = os.path.join(data_dir, 'images', _split)

        # Load if cached data exist
        self.cached_data_file = os.path.join(data_dir, 'cached_nlvr2_data', f'{_split}.pkl')
        if os.path.exists(self.cached_data_file):
            with open(self.cached_data_file, 'rb') as f:
                self.data = pickle.load(open(self.cached_data_file, 'rb'))
        else:
            annotations_file = os.path.join(data_dir, 'data', f'{_split}.json')

            self.data = []
            # https://github.com/facebookresearch/vilbert-multi-task/blob/main/vilbert/datasets/nlvr2_dataset.py
            with jsonlines.open(annotations_file) as reader:
                for annotation in reader:
                    # logger.info(annotation)
                    example = {}
                    example["id"] = annotation["identifier"]
                    example["image_id_0"] = os.path.join(self.image_dir, (
                        "-".join(annotation["identifier"].split("-")[:-1]) + "-img0.png"
                    ))
                    example["image_id_1"] = os.path.join(self.image_dir, (
                        "-".join(annotation["identifier"].split("-")[:-1]) + "-img1.png"
                    ))
                    example["sentence"] = str(annotation["sentence"])
                    example["labels"] = 0 if str(annotation["label"]) == "False" else 1
                    self.data.append(example)

            with open(self.cached_data_file, 'wb') as f:
                pickle.dump(self.data, f)

        self.n_examples = len(self.data)
        logger.info("Loaded NLVRv2 {} dataset, with {} examples".format(split, self.n_examples))
        self.pil_transform = T.Resize(size=384, max_size=640)

    def get_pil_image(self, image_fn):
        image = Image.open(image_fn)
        image = image.convert('RGB')
        if min(list(image.size)) > 384:
            image = self.pil_transform(image)
        return image

    def __len__(self):
        return self.n_examples

    def __getitem__(self, index: int):

        """
        Args:
        index : index of element in self.data to return as data instance

        Returns:
        dictionary containing inputs and targets for model to do NLVR
        """

        example = self.data[index]
        img1 = self.get_pil_image(example["image_id_0"])
        img2 = self.get_pil_image(example["image_id_1"])
        image = [img1, img2]

        return {'text': example["sentence"], 
                'image': image, 
                'label': example["labels"]}

    def convert_to_low_shot(self, num_shots_per_class: int):
        """
        Args:
        num_shots_per_class: int, denoting number of examples for each output label in low-shot setting
        """

        assert self.split == 'train'
        logger.info("Converting NLVR2 train split into low-shot dataset, with {} examples per class...".format(num_shots_per_class))
        new_data = []
        for i in range(self.num_labels):
            i_examples = [d for d in self.data if d['labels'] == i]
            low_shot_examples = random.sample(i_examples, num_shots_per_class)
            new_data.extend(low_shot_examples)
        self.data = new_data
        self.n_examples = len(self.data)

        logger.info("Converted into low-shot dataset, with {} examples".format(self.n_examples))

def nlvr2_batch_collate(batch: List[Dict], 
                        visual_input_type: str):

    """
    Collates each model input for all batch items into a single model input (e.g. converts a list of input_ids into a matrix of size (batch_size, max_len))

    Args:
    batch - list of batch items, each item being a dictionary returned by Dataset's __getitem__ method
    visual_input_type: string which specifies the type of visual input

    Returns:
    Dictionary containing batched inputs and outputs
    """

    assert visual_input_type == 'pil-image'
    texts = [x['text'] for x in batch]
    pil_objs = [x['image'] for x in batch]
    labels = [x['label'] for x in batch]

    return {'raw_texts': texts, 
            'images': pil_objs, 
            'labels': torch.LongTensor(labels)}

def build_nlvr2_dataloader(args, 
                           data_dir: str, 
                           split: str, 
                           visual_input_type: str,
                           **kwargs) -> torch.utils.data.DataLoader:

    """
    Creates the NLVR2 Dataloader, which gives batches of NLVR2 inputs and outputs

    Args:
    data_dir : path containing NLVR questions and annotations.
    split: either train/val split
    visual_input_type: format of visual input to model

    Returns:
    DataLoader object
    """

    logger.info("Creating NLVR2 {} dataloader with batch size of {}".format(split, int(args.batch_size/2)))

    if visual_input_type != "pil-image":
        raise NotImplementedError("Have not implemented other inputs for NLVR2 images!")

    dataset = NLVR2Dataset(data_dir, split, **kwargs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers = args.num_workers,
        batch_size = int(args.batch_size/2),
        shuffle = (split=='train'),
        collate_fn = lambda x: nlvr2_batch_collate(x, visual_input_type)
        )
    return dataloader

    
'''
if __name__ == '__main__':

    class Args:
        def __init__(self):
            self.batch_size = 4
            self.num_workers = 2
            self.visual_input_type = 'pil-image'
    
    args = Args()
    data_dir          = '/data/datasets/MCL/nlvr2/'
    split             = 'val' #'train' 
    
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    nlvr_dataloader = build_nlvr2_dataloader(args, data_dir,'val', args.visual_input_type, tokenizer=tokenizer)

    for batch in nlvr_dataloader:
        pdb.set_trace() 
'''
