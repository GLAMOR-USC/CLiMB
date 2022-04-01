
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

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset

from transformers import BertTokenizer

from PIL import Image
from utils.image_utils import resize_image

from data.image_datasets.flickr30kimages_dataset import Flickr30KImagesDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class SnliVEDataset(Dataset):

    def __init__(self, data_dir, images_dataset, split, tokenizer, visual_mode='raw'):

        self.data_dir = data_dir
        self.images_dataset = images_dataset
        self.image_dir = os.path.join(data_dir, 'flickr30k_images')
        self.split = split
        self.tokenizer = tokenizer
        self.visual_mode = visual_mode

        self.annotations_file = os.path.join(data_dir, 'snli_ve_{}.jsonl'.format(split))
        self.categories = ['entailment', 'contradiction', 'neutral']
        self.cat2label = {cat: i for i, cat in enumerate(self.categories)}

        self.cached_data_file = os.path.join(data_dir, 'cached_vqa_data', 'snli-ve_{}.pkl'.format(split))
        if os.path.isfile(self.cached_data_file):
            self.data = pkl.load(open(self.cached_data_file, 'rb'))
        else:
            self.data = []
            json_lines = jsonlines.open(self.annotations_file)
            for line in json_lines:
                image_id = int(line['Flickr30K_ID'])
                hypothesis = str(line['sentence2'])
                gold_label = self.cat2label[line['gold_label']]

                tokens = self.tokenizer.tokenize(hypothesis)
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                doc = {'image_id': image_id,
                        'hypothesis': hypothesis,
                        'hypothesis_input_ids': input_ids,
                        'label': gold_label}
                self.data.append(doc)

            pkl.dump(self.data, open(self.cached_data_file, 'wb'))

        logger.info("Loaded SNLI-VE {} dataset, with {} examples".format(self.split, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        example = self.data[index]

        # Tokenizer the input hypothesis 
        hypothesis = example['hypothesis']
        input_ids = example['hypothesis_input_ids']

        # Get the image tensor from ImageDataset
        image_id = example['image_id']
        image = self.images_dataset.get_image_data(image_id, self.visual_mode)

        label = example['label']

        return hypothesis, input_ids, image, label

def snlive_batch_collate(batch, visual_mode):

    #pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]   # should be 0, but doing this anyway
    pad_token = 0   # tokenizer.pad_token_id

    # Pad the text inputs
    hypotheses = [x[0] for x in batch]
    input_ids = [x[1] for x in batch]
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

    # Create labels tensor
    labels = [x[3] for x in batch]
    labels = torch.tensor(labels, dtype=torch.long)

    # Stack the image tensors, doing padding if necessary for the sequence of region features
    image_tensors = [x[2] for x in batch]
    if visual_mode == 'pil-image':
        images = image_tensors                                          # Not actually tensors for this option, list of PIL.Image objects
    if visual_mode == 'raw':
        images = torch.stack(image_tensors, dim=0)               # Stacks individual raw image tensors to give (B, 3, W, H) tensor
    elif visual_mode == 'fast-rcnn':
        max_len = max([t.shape[0] for t in image_tensors])
        image_tensors_padded = []
        for i in range(len(image_tensors)):
            padding_tensor = torch.zeros(max_len-image_tensors[i].shape[0], image_tensors[i].shape[1])
            padded_tensor = torch.cat((image_tensors[i], padding_tensor), dim=0)
            assert padded_tensor.shape[0] == max_len
            image_tensors_padded.append(padded_tensor)
        images = torch.stack(image_tensors_padded, dim=0)        # Pads region features with 0 vectors to give (B, R, hv) tensor

    return {'raw_texts': hypotheses,
            'input_ids': input_ids,
            'attn_mask': attn_mask,
            'images': images,
            'labels': labels}

def build_snli_ve_dataloader(args, data_dir, images_dataset, split, tokenizer, visual_mode):

    batch_size = args.batch_size
    shuffle = True if split == 'train' else False

    logger.info("Creating SNLI-VE {} dataloader with batch size of {}".format(split, batch_size))

    dataset = SnliVEDataset(data_dir, images_dataset, split, tokenizer, visual_mode)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: snlive_batch_collate(x, visual_mode))

    return dataloader

if __name__ == '__main__':
    data_dir = '/data/datasets/MCL/snli-ve/'
    class Args:
        def __init__(self):
            self.batch_size = 4
            self.shuffle = True
            self.num_workers = 2
            self.visual_mode = 'pil-image'
    args = Args()

    images_dataset = Flickr30KImagesDataset('/data/datasets/MCL/flickr30k/')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    snli_ve_dataloader = build_snli_ve_dataloader(args, data_dir, images_dataset, 'train', tokenizer, args.visual_mode)

    for batch in snli_ve_dataloader:
        pdb.set_trace() 
