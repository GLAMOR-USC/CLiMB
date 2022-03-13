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

sys.path.insert(0, '.')

import numpy as np
import torch
from tqdm import tqdm
import wandb

from transformers import BertTokenizer

from modeling.vilt_modeling_debug import load_vilt

from configs.model_configs import model_configs
from configs.task_configs import task_configs
from utils.seed_utils import set_seed
from data.visionlanguage_datasets.nlvr2_dataset import build_nlvr2_dataloader

logger = logging.getLogger(__name__)

device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

class Args:
    def __init__(self):
        self.batch_size = 64
        self.shuffle = True
        self.num_workers = 2
        self.encoder_name = 'vilt'
        self.pretrained_model_name = 'dandelin/vilt-b32-finetuned-nlvr2'
        self.ordered_cl_tasks = ['nlvr2']
        self.seed = 42



def eval_nlvr2(args, model, task_configs, model_config, device):
    # no further training here, eval the pretrained model
    nlvr_config = task_configs['nlvr2']
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
            output = model(**inputs)
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
    model = load_vilt(args.pretrained_model_name, device)
    model.to(device)

    eval_nlvr2(args, model, task_configs, model_config, device)
