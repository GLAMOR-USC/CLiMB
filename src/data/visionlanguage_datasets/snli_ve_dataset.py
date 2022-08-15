import sys
import os
import time
import json
import logging
import random
import glob
import base64
from tqdm import tqdm
from collections import defaultdict
import pickle as pkl
import pdb
import jsonlines
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset

from transformers import BertTokenizer

from PIL import Image
from utils.image_utils import resize_image

from data.image_datasets.flickr30kimages_dataset import Flickr30KImagesDataset
from data.image_collation import image_collate

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class SnliVEDataset(Dataset):

    def __init__(self, 
                 data_dir: str, 
                 images_dataset: Flickr30KImagesDataset, 
                 split: str, 
                 **kwargs):

        """
        Initiates the SnliVEDataset - loads all the questions (and converts to input IDs using the tokenizer, if provided) 
        and answers (including converting each to a numeric label, and a score based on occurence from annotators)
        Every item in self.data corresponds to a single VE hypothesis and corresponding image

        Args:
        data_dir : path containing SNLI-VE hypotheses and annotations.
        images_dataset : instance of Flickr30KImagesDataset, that is used to retrieve the Flickr30K image for each question
        split: either train/val split


        Returns:
        Loads all annotations into self.data, where each item is a single SNLI-VE pair
        """

        self.data_dir = data_dir
        self.images_dataset = images_dataset
        self.image_dir = os.path.join(data_dir, 'flickr30k_images')
        self.split = split
        self.tokenizer = kwargs['tokenizer'] if 'tokenizer' in kwargs else None

        self.annotations_file = os.path.join(data_dir, 'snli_ve_{}.jsonl'.format(split))
        self.categories = ['entailment', 'contradiction', 'neutral']
        self.cat2label = {cat: i for i, cat in enumerate(self.categories)}
        self.num_labels = len(self.categories)

        self.cached_data_file = os.path.join(data_dir, 'cached_ve_data', 'snli-ve_{}.pkl'.format(split))
        if os.path.isfile(self.cached_data_file):
            self.data = pkl.load(open(self.cached_data_file, 'rb'))
        else:
            self.data = []
            json_lines = jsonlines.open(self.annotations_file)
            for line in tqdm(json_lines):
                image_id = int(line['Flickr30K_ID'])
                hypothesis = str(line['sentence2'])
                gold_label = self.cat2label[line['gold_label']]

                if self.tokenizer is not None:
                    tokens = self.tokenizer.tokenize(hypothesis)
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                else:
                    input_ids = []

                doc = {'image_id': image_id,
                        'hypothesis': hypothesis,
                        'hypothesis_input_ids': input_ids,
                        'label': gold_label}
                self.data.append(doc)

            pkl.dump(self.data, open(self.cached_data_file, 'wb'))

        logger.info("Loaded SNLI-VE {} dataset, with {} examples".format(self.split, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        """
        Args:
        index : index of element in self.data to return as data instance

        Returns:
        dictionary containing inputs and targets for model to do SNLI-VE
        """

        example = self.data[index]

        # Tokenizer the input hypothesis 
        hypothesis = example['hypothesis']
        input_ids = example['hypothesis_input_ids']

        # Get the image tensor from ImageDataset
        image_id = example['image_id']
        image = self.images_dataset.get_image_data(image_id)

        label = example['label']

        return {'hypothesis': hypothesis, 
                'input_ids': input_ids, 
                'image': image, 
                'label': label
                }


    def convert_to_low_shot(self, num_shots_per_class: int):
        """
        Args:
        num_shots_per_class: int, denoting number of examples for each output label in low-shot setting
        """

        assert self.split == 'train'
        logger.info("Converting SNLI-VE train split into low-shot dataset, with {} examples per class...".format(num_shots_per_class))
        new_data = []
        for i in range(self.num_labels):
            i_examples = [d for d in self.data if d['label'] == i]
            low_shot_examples = random.sample(i_examples, num_shots_per_class)
            new_data.extend(low_shot_examples)
        self.data = new_data
        self.n_examples = len(self.data)
        logger.info("Converted into low-shot dataset, with {} examples".format(self.n_examples))


def snlive_batch_collate(batch: List[Dict], 
                        visual_input_type: str):

    """
    Collates each model input for all batch items into a single model input (e.g. converts a list of input_ids into a matrix of size (batch_size, max_len))

    Args:
    batch - list of batch items, each item being a dictionary returned by Dataset's __getitem__ method
    visual_input_type: string which specifies the type of visual input

    Returns:
    Dictionary containing batched inputs and outputs
    """

    pad_token = 0   # tokenizer.pad_token_id

    # Pad the text inputs
    hypotheses = [x['hypothesis'] for x in batch]
    input_ids = [x['input_ids'] for x in batch]
    max_len = max([len(x) for x in input_ids])
    input_ids_padded = []
    attn_masks = []
    for i in range(len(input_ids)):
        ids_padded = input_ids[i] + [pad_token]*(max_len - len(input_ids[i]))
        attn_mask = [1]*len(input_ids[i]) + [0]*(max_len - len(input_ids[i]))

        input_ids_padded.append(ids_padded)
        attn_masks.append(attn_mask)
    input_ids = torch.tensor(input_ids_padded, dtype=torch.long)
    attn_mask = torch.tensor(attn_masks, dtype=torch.long)

    # Stack the target tensors
    # Create labels tensor
    labels = [x['label'] for x in batch]
    labels = torch.tensor(labels, dtype=torch.long)

    # Depending on the visual_input_type variable, process the images accordingly
    images = [x['image'] for x in batch]
    images = image_collate(images, visual_input_type)

    return {'raw_texts': hypotheses,
            'input_ids': input_ids,
            'attn_mask': attn_mask,
            'images': images,
            'labels': labels}

def build_snli_ve_dataloader(args, 
                             data_dir: str, 
                             images_dataset: Flickr30KImagesDataset, 
                             split: str, 
                             visual_input_type: str,
                             **kwargs) -> torch.utils.data.DataLoader:

    """
    Creates the SNLI-VE Dataloader, which gives batches of SNLI-VE inputs and outputs

    Args:
    args
    data_dir : path containing SNLI-VE hypotheses and annotations.
    images_dataset : instance of Flickr30KImagesDataset, that is used to retrieve the Flickr30K image for each question
    split: either train/val split
    visual_input_type: format of visual input to model

    Returns:
    DataLoader object
    """


    batch_size = args.batch_size
    shuffle = True if split == 'train' else False

    logger.info("Creating SNLI-VE {} dataloader with batch size of {}".format(split, batch_size))

    dataset = SnliVEDataset(data_dir, images_dataset, split, **kwargs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: snlive_batch_collate(x, visual_input_type))

    return dataloader

if __name__ == '__main__':
    data_dir = '/data/datasets/MCL/snli-ve/'
    class Args:
        def __init__(self):
            self.batch_size = 4
            self.shuffle = True
            self.num_workers = 2
            self.visual_input_type = 'pil-image'
    args = Args()

    images_dataset = Flickr30KImagesDataset('/data/datasets/MCL/flickr30k/', args.visual_input_type)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    snli_ve_dataloader = build_snli_ve_dataloader(args, data_dir, images_dataset, 'train', args.visual_input_type, tokenizer=tokenizer)

    for batch in snli_ve_dataloader:
        pdb.set_trace() 
