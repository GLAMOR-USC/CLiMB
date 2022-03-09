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

from modeling import load_encoder_map

from configs.model_configs import model_configs
from configs.task_configs import task_configs, SUPPORTED_VL_TASKS
from utils.seed_utils import set_seed

logger = logging.getLogger(__name__)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class Args:
    def __init__(self):
        self.batch_size = 32
        self.shuffle = True
        self.num_workers = 2
        self.encoder_name = 'vilt'
        self.pretrained_model_name = 'dandelin/vilt-b32-mlm' #TODO
        self.ordered_cl_tasks = ['vqa']
        self.seed = 42

        self.output_dir = '/data/experiments/MCL/'
        self.cl_algorithm = 'singletask_ft'     # Must not contain hyphens
        self.wandb_project_name = 'vl-cl'

if __name__ == '__main__':
    args = Args()

    experiment_name = '{}-{}'.format(args.encoder_name, args.cl_algorithm)
    for i, task_key in enumerate(args.ordered_cl_tasks):
        experiment_name = '{}-task{}_{}'.format(experiment_name, i, task_key)
    output_dir = os.path.join(args.output_dir, experiment_name)
    results_file = os.path.join(output_dir, 'results.json')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    logger.info('W&B project name: {}'.format(args.wandb_project_name))
    wandb.init(project=args.wandb_project_name,
        name=experiment_name,
        entity='tejas1995')

    set_seed(args)
    # Load the correct Encoder model, based on encoder_name argument
    model_config = model_configs[args.encoder_name]
    load_encoder_method = load_encoder_map[args.encoder_name]
    encoder = load_encoder_method(args.pretrained_model_name, device)

    # Ensure all the tasks for continual learning are supported VL tasks
    for task_key in args.ordered_cl_tasks:
        assert task_key in SUPPORTED_VL_TASKS

    results = []
    for task_num, task_key in enumerate(args.ordered_cl_tasks):
        # Load the correct training method for current CL task, and call the training method

        task_name = task_configs[task_key]['task_name']
        logger.info("-"*100)
        logger.info("Training {} model on task #{}: {}".format(args.encoder_name, task_num+1, task_name))
        train_method = task_configs[task_key]['train_method']
        best_eval_score, best_model = train_method(args, encoder, task_configs, model_config, tokenizer, device)

        logger.info("Best {} evaluation score = {:.2f}, after epoch {}".format(task_name, best_eval_score, best_model['epoch']+1))

        # Save best model checkpoint, and separately save the models' Encoder object
        logger.info("Saving best model and encoder checkpoint after {} training".format(task_name))
        task_output_dir = os.path.join(output_dir, 'checkpoints', 'task{}_{}'.format(task_num, task_key))
        if not os.path.isdir(task_output_dir):
            os.makedirs(task_output_dir)
        best_task_model = best_model['model']
        torch.save(best_task_model.state_dict(), os.path.join(task_output_dir, 'model'))
        torch.save(best_task_model.get_encoder().state_dict(), os.path.join(task_output_dir, 'encoder'))
        logger.info("Saved checkpoint!")

        # Save CL results so far
        task_results = {
            'task_num': task_num,
            'task_key': task_key,
            'best_score': best_eval_score,
            'best_epoch': best_model['epoch']
        }
        results.append(task_results)
        json.dump(results, open(results_file, 'w'))
        logger.info("Saved continual learning results so far!")
