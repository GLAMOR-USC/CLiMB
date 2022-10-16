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

from PIL import Image

from data.image_collation import image_collate

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Skyler', 'Frankie', 'Pat', 'Quinn', 'Morgan', 'Finley', 'Harley', 'Robbie', 'Sidney', 'Tommie',
                        'Ashley', 'Carter', 'Adrian', 'Clarke', 'Logan', 'Mickey', 'Nicky', 'Parker', 'Tyler',
                        'Reese', 'Charlie', 'Austin', 'Denver', 'Emerson', 'Tatum', 'Dallas', 'Haven', 'Jordan',
                        'Robin', 'Rory', 'Bellamy', 'Salem', 'Sutton', 'Gray', 'Shae', 'Kyle', 'Alex', 'Ryan',
                        'Cameron', 'Dakota']


def process_list(mytext, objects):
    ## Read file with the name of the color per object
    
    ## processing the text
    text = ''
    for element in mytext:
        #print(element)
        if(type(element) == list):     #### If it's a list we need to process each object 
            for subelement in element:
                if(objects[int(subelement)] == 'person'):
                    temporal_text = GENDER_NEUTRAL_NAMES[int(subelement)]
                else:
                    temporal_text = 'the gray ' + str(objects[int(subelement)]).strip()
        elif(type(element) == int):
            if(objects[int(element)] == 'person'):
                temporal_text = GENDER_NEUTRAL_NAMES[int(subelement)]
            else:
                temporal_text = 'the gray ' + str(objects[int(subelement)])
        else:
            temporal_text = element
        text += temporal_text + ' '
    #print('text: ', text)
    return text

class VCRDataset(Dataset):

    def __init__(self, 
                 data_dir: str, 
                 split: str, 
                 task_type='qa',
                 **kwargs):

        """
        Initiates the VCRDataset - loads all the questions and answers, concatenates each question and answer into a choice text
        (and converts to input IDs using the tokenizer, if provided) and stores all 4 choice texts and label 
        Every item in self.data corresponds to a single VCR input

        Args:
        data_dir : path containing VCR questions and annotations
        split: either train/val/test split
        task_type: either 'qa' or 'qar', depending on if we do Q->A or QA->R

        Returns:
        Loads all annotations into self.data, where each item is a single VCR input
        """

        self.data_dir = data_dir
        self.images_dataset = self.data_dir + 'draw_images/bbox/'    
            
        self.image_dir = os.path.join(data_dir, 'vcr')
        self.split = split
        self.task_type = task_type
        self.tokenizer = kwargs['tokenizer'] if 'tokenizer' in kwargs else None

        self.annotations_file = os.path.join(data_dir, 'annotation/{}.jsonl'.format(split))

        self.cached_data_file = os.path.join(data_dir, 'cached_vcr_data', 'vcr_'+ str(task_type) + '_' + '{}.pkl'.format(split))
        if os.path.exists(self.cached_data_file):
            self.data = pkl.load(open(self.cached_data_file, 'rb'))
        else:
            self.data = []
            json_lines = jsonlines.open(self.annotations_file)
            count = 0
            for line in tqdm(json_lines):
                
                image_path = os.path.join('drawn_images/bbox/' + str(split) + '/' + str(task_type)+ '/' + str(line['annot_id']) +'.jpg')  ## train-0, train-1, train-2
                multichoice_texts = []
                objects = line['objects']   ### objects
                
                question = process_list(line['question'], objects)  ### question
                if(task_type == 'qa'):
                    ### answers:     question + ' [SEP] ' + answer
                    for answer in line['answer_choices']:                        
                        answer1 = process_list(answer, objects)
                        text = question + ' [SEP] ' + answer1
                        multichoice_texts.append(text)
                    label = int(line['answer_label']) ##number

                else:
                    ### rationales:  question + '[SEP]' + answer + '[SEP]' + rationale
                    answer  = process_list( line['answer_choices'][int(line['answer_label'])], objects)
                    for rationale in line['rationale_choices']:
                        rationale1 = process_list(rationale, objects)
                        text = question + ' [SEP] ' + answer + ' [SEP] ' + rationale1
                        multichoice_texts.append(text)
                    label = int(line['rationale_label']) ##number

                if self.tokenizer is not None:
                    multichoice_tokens = [self.tokenizer.tokenize(text) for text in multichoice_texts]
                    multichoice_input_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in multichoice_tokens]
                else:
                    multichoice_input_ids = []

                doc = {'image_path': image_path,
                        'texts': multichoice_texts,
                        'input_ids': multichoice_input_ids,
                        'label': label}
                self.data.append(doc)
                
            pkl.dump(self.data, open(self.cached_data_file, 'wb'))
        self.n_examples = len(self.data)
        logger.info("Loaded VCR-{} {} dataset, with {} examples".format(self.task_type, self.split, len(self.data)))

    def __len__(self):
        return self.n_examples

    def __getitem__(self, index: int):

        """
        Args:
        index : index of element in self.data to return as data instance

        Returns:
        dictionary containing inputs and targets for model to do VCR
        """

        example = self.data[index]
        
        image_fn     = os.path.join(self.data_dir, example['image_path'])
        pil_transform = T.Resize(size=384, max_size=640)
        image = Image.open(image_fn)
        image = image.convert('RGB')
        if min(list(image.size)) > 384:
            image = pil_transform(image)

        texts    = example['texts']
        label   = example['label']

        return  {'texts': texts, 
                 'image': image, 
                 'label': label
                 }

    def convert_to_low_shot(self, low_shot_percentage: float):
        """
        Args:
        low_shot_percentage: float between 0 and 1, telling what % of full data to retain for low-shot setting
        """

        assert self.split == 'train'
        logger.info("Converting VCR train split into low-shot dataset, with {:.2f}% training samples...".format(low_shot_percentage*100.0))
        n_low_shot_examples = int(low_shot_percentage*self.n_examples)

        new_data = random.sample(self.data, n_low_shot_examples)
        self.data = new_data
        self.n_examples = len(self.data)

        logger.info("Converted into low-shot dataset, with {} examples".format(self.n_examples))

def vcr_batch_collate(batch: List[Dict], 
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
    texts = [x['texts'] for x in batch]
    pil_objs = [x['image'] for x in batch]
    labels = [x['label'] for x in batch]

    return {'raw_texts': texts,
            'images': pil_objs,
            'labels': torch.LongTensor(labels)}

def build_vcr_dataloader(args, 
                         data_dir: str, 
                         split: str, 
                         task_type: str, 
                         visual_input_type: str,
                         **kwargs) -> torch.utils.data.DataLoader:

    """
    Creates the VCR Dataloader, which gives batches of VCR inputs and outputs

    Args:
    data_dir : path containing VCR questions and annotations.
    split: either train/val split
    task_type: either 'qa' or 'qar', depending on if we do Q->A or QA->R
    visual_input_type: format of visual input to model

    Returns:
    DataLoader object
    """

    batch_size = int(args.batch_size/4)
    shuffle = True if split == 'train' else False

    assert visual_input_type == 'pil-image'     # VCR not supported for other visual inputs yet!

    logger.info("Creating VCR {} dataloader with batch size of {}".format(split, batch_size))

    dataset = VCRDataset(data_dir, split, task_type, **kwargs)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: vcr_batch_collate(x, visual_input_type))
    return dataloader

    
if __name__ == '__main__':

    class Args:
        def __init__(self):
            self.batch_size = 16
            self.num_workers = 2
            self.visual_input_type = 'pil-image'
    
    args = Args()
    data_dir          = '/data/datasets/MCL/vcr/'
    #annotation_dir    = '/data/datasets/MCL/vcr/annotation/'
    split             = 'val' #'train' 
    text    = ['Why', 'is', [0], 'smiling', 'at', [1], '?']
    objects = ['person', 'person', 'bottle']
    #process_list(text, objects)

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #vcr.VCRDataset(data_dir, split, tokenizer, task_type='qa')

    vcr_train_dataloader  = build_vcr_dataloader(args, data_dir, split= 'train', tokenizer = tokenizer, task_type = 'qa', visual_input_type=args.visual_input_type)
    vcr_val_dataloader  = build_vcr_dataloader(args, data_dir, split= 'val',  tokenizer = tokenizer, task_type = 'qa', visual_input_type=args.visual_input_type)

    vcr_train_dataloader  = build_vcr_dataloader(args, data_dir, split= 'train', task_type = 'qar', visual_input_type=args.visual_input_type)
    vcr_val_dataloader  = build_vcr_dataloader(args, data_dir, split= 'val', task_type = 'qar', visual_input_type=args.visual_input_type)

    for batch in vcr_val_dataloader:
        pdb.set_trace()