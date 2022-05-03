import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
import math
import shutil
import pickle as pkl
import copy
import yaml
import pdb
import numpy as np
import torch
from tqdm import tqdm

from modeling.vilt_modeling import ViltForImageTextClassification
from data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset
from configs.model_configs import model_configs
from configs.task_configs import task_configs
from utils.seed_utils import set_seed
from data.visionlanguage_datasets.nlvr2_dataset import build_nlvr2_dataloader
from data.visionlanguage_datasets.vqa_dataset import build_vqa_dataloader
from train.train_vqa import compute_score_with_logits

from transformers import BertTokenizer

sys.path.insert(0, '.')

logger = logging.getLogger(__name__)

device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

class Args:
    def __init__(self):
        self.task_name = 'vqa'
        self.batch_size = 64
        self.shuffle = True
        self.num_workers = 2
        self.encoder_name = 'vilt'
        self.seed = 42


def eval_vqa(args, model, vqa_config, model_config, device):
    data_dir = vqa_config['data_dir']
    visual_mode = model_config['visual_mode']
    batch2inputs_converter = model_config['batch2inputs_converter']


    # Load COCO Images dataset for image data backbone
    images_source = vqa_config['images_source']
    mscoco_config = task_configs[images_source]
    images_dataset = MSCOCOImagesDataset(mscoco_config['data_dir'])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vqa_val_dataloader = build_vqa_dataloader(args=args,
                                          data_dir=data_dir,
                                          images_dataset=images_dataset,
                                          split='val',
                                          tokenizer=tokenizer,
                                          visual_mode=visual_mode)
    model.eval()
    eval_score = 0

    for step, batch in enumerate(tqdm(vqa_val_dataloader, desc='Evaluating on VQA val set')):
        inputs = batch2inputs_converter(batch)
        target = batch['target_scores'].to(device)

        with torch.no_grad():
            output = model(**inputs)
        logits = output[1]

        answer_scores = compute_score_with_logits(logits, target, device)
        batch_scores = torch.sum(answer_scores, 1)

        eval_score += batch_scores.sum().item()

    eval_score = eval_score/len(vqa_val_dataloader.dataset)*100.0
    logger.info(f'Eval_acc: {eval_score:.3f}')

    return eval_score


def eval_nlvr2(args, model, nlvr_config, model_config, device):
    # no further training here, eval the pretrained model
    data_dir = nlvr_config['data_dir']
    visual_mode = model_config['visual_mode']
    batch2inputs_converter = model_config['batch2inputs_converter']


    val_dataloader = build_nlvr2_dataloader(args=args,
                                          data_dir=data_dir,
                                          split='val',
                                          visual_mode=visual_mode)

    model.eval()
    eval_score = 0
    for step, batch in enumerate(tqdm(val_dataloader, desc='Evaluating on NLVR2 val set')):
        inputs = batch2inputs_converter(batch)
        with torch.no_grad():
            output = model.fwd_multi_imgs(**inputs)
            logits = output[1]

        batch_scores = (logits.argmax(-1).cpu() == batch['labels'])
        eval_score += batch_scores.sum().item()

    eval_score = eval_score/len(val_dataloader.dataset)*100.0
    logger.info(f'Eval_acc: {eval_score:.3f}')


if __name__ == '__main__':
    args = Args()

    set_seed(args)
    # Load pretrained
    model_config = model_configs[args.encoder_name]
    ckpt_path = '/data/experiments/MCL/vilt-singletask_ft-task0_nlvr2/nlvr_best_model.tar'
    checkpoint = torch.load(ckpt_path)

    task_config = task_configs[args.task_name]
    num_labels = task_config['num_labels']
    num_images = 2 if args.task_name == 'nlvr2' else 1
    model = ViltForImageTextClassification(checkpoint['encoder'], 768, num_labels, num_images)

    if args.task_name in ckpt_path: # model finetuned on the same task
        model.load_state_dict(checkpoint['model'])
    else: # model finetuned on a different task; thus, the classifier has different dimensions
        model_dict = model.state_dict()
        ckpt_dict = checkpoint['model']
        for k in ckpt_dict.keys():
            if 'clf_layer' not in k:
                model_dict[k] = ckpt_dict[k]
                #print(k, 'loaded!')
    model.to(device)

    #eval_nlvr2(args, model, task_config, model_config, device)
    eval_vqa(args, model, task_config, model_config, device)
