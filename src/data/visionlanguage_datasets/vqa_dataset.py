
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

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset

from transformers import BertTokenizer

from PIL import Image
from utils.image_utils import resize_image
from utils.vqa_utils import get_score, target_tensor

from data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class VQADataset(Dataset):

    def __init__(self, data_dir, images_dataset, split, tokenizer, visual_mode='raw'):

        self.images_dataset = images_dataset
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.visual_mode = visual_mode

        self.annotations_file = os.path.join(data_dir, 'v2_mscoco_{}2014_annotations.json'.format(split))
        self.questions_file = os.path.join(data_dir, 'v2_OpenEnded_mscoco_{}2014_questions.json'.format(split))
        self.ans2label_file = os.path.join(data_dir, 'ans2label.pkl'.format(split)) # Where does this file come from?

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
                tokens = self.tokenizer.tokenize(question)
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

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

        logger.info("Loaded VQAv2 {} dataset, with {} examples".format(self.split, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        example = self.data[index]
        question_id = example['question_id']

        # Tokenizer the input question 
        question = example['question']
        input_ids = example['question_input_ids']

        # Get the image tensor from ImageDataset
        image_id = example['image_id']
        image = self.images_dataset.get_image_data(image_id, self.visual_mode)

        labels = example['labels']
        scores = example['scores']

        return question, input_ids, image, labels, scores, question_id

def batch_collate(batch, tokenizer, visual_mode, num_labels):

    #pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]   # should be 0, but doing this anyway
    pad_token = 0   # tokenizer.pad_token_id

    # Pad the text inputs
    # Do we need to set a global MAX_LEN to clip the super long questions? (this really depends on the dataset)
    questions = [x[0] for x in batch]
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

    # Create the target tensor using scores
    batch_scores = []
    batch_labels = []
    for x in batch:
        labels, scores = x[3], x[4]
        target_scores = target_tensor(num_labels, labels, scores)
        batch_scores.append(target_scores)
        batch_labels.append(labels)
    batch_scores = torch.stack(batch_scores, dim=0)

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

    return {'raw_texts': questions,
            'input_ids': input_ids,
            'attn_mask': attn_mask,
            'images': images,
            'target_scores': batch_scores,
            'labels': batch_labels}

def build_vqa_dataloader(args, data_dir, images_dataset, split, tokenizer, visual_mode):

    batch_size = args.batch_size
    shuffle = True if split == 'train' else False

    logger.info("Creating VQAv2 {} dataloader with batch size of {}".format(split, batch_size))

    dataset = VQADataset(data_dir, images_dataset, split, tokenizer, visual_mode)
    num_labels = dataset.num_labels
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: batch_collate(x, tokenizer, visual_mode, num_labels))
    return dataloader

if __name__ == '__main__':
    data_dir = '/data/datasets/MCL/vqav2/'
    #dataset = VQADataset(data_dir, None, 'train', None)
    class Args:
        def __init__(self):
            self.batch_size = 4
            self.shuffle = True
            self.num_workers = 2
            self.visual_mode = 'pil-image'
    args = Args()

    images_dataset = MSCOCOImagesDataset('/data/datasets/MCL/ms-coco/')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vqa_dataloader = build_vqa_dataloader(args, data_dir, images_dataset, 'val', tokenizer, args.visual_mode)

    for batch in vqa_dataloader:
        pdb.set_trace() 
