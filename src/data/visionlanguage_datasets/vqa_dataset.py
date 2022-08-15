
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
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset

from PIL import Image
from utils.image_utils import resize_image
from utils.vqa_utils import get_score, target_tensor

from data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset
from data.image_collation import image_collate

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class VQADataset(Dataset):

    def __init__(self, 
                 data_dir: str, 
                 images_dataset: MSCOCOImagesDataset, 
                 split: str, 
                 **kwargs):

        """
        Initiates the VQADataset - loads all the questions (and converts to input IDs using the tokenizer, if provided) 
        and answers (including converting each to a numeric label, and a score based on occurence from annotators)
        Every item in self.data corresponds to a single QA pair, with a corresponding image

        Args:
        data_dir : path containing VQA questions and annotations. Also contains mapping from each answer in set of possible answers to a numerical label
        images_dataset : instance of MSCOCOImagesDataset, that is used to retrieve the MS-COCO image for each question
        split: either train/val split

        Returns:
        Loads all annotations into self.data, where each item is a single VQA pair
        """

        self.images_dataset = images_dataset
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = kwargs['tokenizer'] if 'tokenizer' in kwargs else None

        self.annotations_file = os.path.join(data_dir, 'v2_mscoco_{}2014_annotations.json'.format(split))
        self.questions_file = os.path.join(data_dir, 'v2_OpenEnded_mscoco_{}2014_questions.json'.format(split))
        self.ans2label_file = os.path.join(data_dir, 'ans2label.pkl'.format(split))

        # Load mapping from answers to labels
        self.ans2label = pkl.load(open(self.ans2label_file, 'rb'))
        self.label2ans = {v: k for k, v in self.ans2label.items()}
        self.num_labels = len(self.label2ans)


        self.cached_data_file = os.path.join(data_dir, 'cached_vqa_data', 'vqa_{}.pkl'.format(split))
        if os.path.isfile(self.cached_data_file):
            # Load cached data
            self.data = pkl.load(open(self.cached_data_file, 'rb'))

        else:
            # Create map from question id to question
            questions = json.load(open(self.questions_file))['questions']
            qid2qdata = {x['question_id']: x for x in questions}

            # Create data for each annotation
            annotations = json.load(open(self.annotations_file))['annotations']
            self.data = []
            for anno in annotations:
                qid = anno['question_id']
                correct_answer = anno['multiple_choice_answer']
                image_id = anno['image_id']

                # Retrieve the question for this annotation
                qdata = qid2qdata[qid]
                assert qdata['image_id'] == image_id
                question = qdata['question']
                if self.tokenizer is not None:
                    tokens = self.tokenizer.tokenize(question)
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                else:
                    tokens = []
                    input_ids = []

                # Map from each crowdsourced answer to occurrences in annotation
                answers = [a['answer'] for a in anno['answers']]
                answer_count = defaultdict(int)
                for ans in answers:
                    answer_count[ans] += 1

                # Get label and score (0.3/0.6/1) corresponding to each crowdsourced answer
                labels = []
                scores = []
                answers = []
                for answer in answer_count:
                    if answer not in self.ans2label:
                        continue
                    labels.append(self.ans2label[answer])
                    score = get_score(answer_count[answer])
                    scores.append(score)
                    answers.append(answer)

                # Store pre-processed example
                example = {'question_id': qid,
                            'image_id': image_id,
                            'question': question,
                            'question_input_ids': input_ids,
                            'correct_answer': correct_answer,
                            'labels': labels,
                            'answers': answers,
                            'scores': scores}
                self.data.append(example)

            pkl.dump(self.data, open(self.cached_data_file, 'wb'))

        self.n_examples = len(self.data)

        logger.info("Loaded VQAv2 {} dataset, with {} examples".format(self.split, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        """
        Args:
        index : index of element in self.data to return as data instance

        Returns:
        dictionary containing inputs and targets for model to do VQA

        """

        example = self.data[index]
        question_id = example['question_id']

        # Tokenize the input question 
        question = example['question']
        input_ids = example['question_input_ids']

        # Get the image tensor from ImageDataset
        image_id = example['image_id']
        image = self.images_dataset.get_image_data(image_id)

        labels = example['labels']
        scores = example['scores']
        target_scores = target_tensor(self.num_labels, labels, scores)

        return {'question': question, 
                'input_ids': input_ids, 
                'image': image, 
                'labels': labels, 
                'target_scores': target_scores, 
                'question_id': question_id
                }

    def convert_to_low_shot(self, low_shot_percentage: float):
        """
        Args:
        low_shot_percentage: float between 0 and 1, telling what % of full data to retain for low-shot setting
        """

        assert self.split == 'train'
        logger.info("Converting VQA train split into low-shot dataset, with {:.2f}% training samples...".format(low_shot_percentage*100.0))
        n_low_shot_examples = int(low_shot_percentage*self.n_examples)

        new_data = random.sample(self.data, n_low_shot_examples)
        self.data = new_data
        self.n_examples = len(self.data)

        logger.info("Converted into low-shot dataset, with {} examples".format(self.n_examples))

def vqa_batch_collate(batch: List[Dict], 
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
    questions = [x['question'] for x in batch]
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
    batch_labels = [x['labels'] for x in batch]
    batch_scores = [x['target_scores'] for x in batch]
    batch_scores = torch.stack(batch_scores, dim=0)

    # Depending on the visual_input_type variable, process the images accordingly
    images = [x['image'] for x in batch]
    images = image_collate(images, visual_input_type)

    return {'raw_texts': questions,
            'input_ids': input_ids,
            'attn_mask': attn_mask,
            'images': images,
            'target_scores': batch_scores,
            'labels': batch_labels}

def build_vqa_dataloader(args, 
                         data_dir: str, 
                         images_dataset: MSCOCOImagesDataset, 
                         split: str, 
                         visual_input_type: str,
                         **kwargs) -> torch.utils.data.DataLoader:

    """
    Creates the VQA Dataloader, which gives batches of VQA inputs and outputs

    Args:
    data_dir : path containing VQA questions and annotations.
    images_dataset : instance of MSCOCOImagesDataset, that is used to retrieve the MS-COCO image for each question
    split: either train/val split
    visual_input_type: format of visual input to model

    Returns:
    DataLoader object
    """

    batch_size = args.batch_size
    shuffle = True if split == 'train' else False

    logger.info("Creating VQAv2 {} dataloader with batch size of {}".format(split, batch_size))

    dataset = VQADataset(data_dir, images_dataset, split, **kwargs)
    num_labels = dataset.num_labels
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: vqa_batch_collate(x, visual_input_type))
    return dataloader

if __name__ == '__main__':
    data_dir = '/data/datasets/MCL/vqav2/'
    #dataset = VQADataset(data_dir, None, 'train', None)
    class Args:
        def __init__(self):
            self.batch_size = 4
            self.shuffle = True
            self.num_workers = 2
            self.visual_input_type = 'pil-image'
    args = Args()

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    images_dataset = MSCOCOImagesDataset('/data/datasets/MCL/ms-coco/', args.visual_input_type)
    vqa_dataloader = build_vqa_dataloader(args, data_dir, images_dataset, 'val', args.visual_input_type, tokenizer=tokenizer)

    for batch in vqa_dataloader:
        pdb.set_trace() 
