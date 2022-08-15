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
from collections import defaultdict
from typing import List, Dict

sys.path.insert(0, '.')

import numpy as np
import torch
from tqdm import tqdm

from transformers import BertTokenizer

from configs.model_configs import model_configs
from configs.task_configs import task_configs, SUPPORTED_VL_TASKS
from utils.seed_utils import set_seed

logger = logging.getLogger(__name__)

def upstream_knowledge_transfer_eval(args: argparse.Namespace, results_file: str) -> Dict:
    '''
    Computes upstream knowledge transfer for CL tasks

    Arguments:
    args
    results_file: JSON file containing model performances on each task in CL setting

    Returns:
    upstream_knowledge_transfer_dict: Dictionary containing relative gain for each CL task
    '''

    cl_results = json.load(open(results_file))
    assert len(cl_results) == len(args.ordered_cl_tasks)

    logger.info("-"*100)
    upstream_knowledge_transfer_dict = {}
    for task_num, task_results in enumerate(cl_results):
        # Get result for this CL task
        task_key = task_results['task_key']
        assert task_key == args.ordered_cl_tasks[task_num]
        task_name = task_configs[task_key]['task_name']
        cl_task_score = task_results['best_score']

        # Get task score for direct finetuning of pretrained model on this task (without CL)
        singletask_output_dir = os.path.join(args.output_dir, '{}-singletask_ft-task0_{}'.format(args.encoder_name, task_key))
        singletask_results = json.load(open(os.path.join(singletask_output_dir, 'results.json')))
        assert len(singletask_results) == 1
        assert singletask_results[0]['task_key'] == task_key
        singletask_score = singletask_results[0]['best_score']

        # Compute relative gain
        random_score = task_configs[task_key]['random_baseline_score']
        relative_gain = 100.0*(cl_task_score - singletask_score)/(singletask_score - random_score)
        logger.info("Absolute performance on task #{}, {} = {:.2f}%".format(task_num, task_name, cl_task_score))
        logger.info("Relative Gain for task #{}, {} = {:.2f}%".format(task_num, task_name, relative_gain))
        upstream_knowledge_transfer_dict[task_key] = {'relative_gain': relative_gain,
                                                      'cl_task_score': cl_task_score,
                                                      'singletask_score': singletask_score}

    return upstream_knowledge_transfer_dict


def catastrophic_forgetting_eval(args: argparse.Namespace, results_file: str, model, task_trainers: List) -> Dict:
    '''
    For model checkpoint after each CL task, compute forgetting of previous CL tasks

    Arguments:
    args
    results_file: JSON file containing model performances on each upstream task in CL setting
    model: Model that was trained in CL setting on multiple sequential tasks
    task_trainers: List of Trainer classes for each CL task

    Returns:
    catastrophic_forgetting_dict: Dictionary containing catastrophic forgetting results
    '''

    model_config = model_configs[args.encoder_name]

    # Load upstream learning results
    cl_results = json.load(open(results_file))
    assert len(cl_results) == len(args.ordered_cl_tasks)
    output_dir = os.path.dirname(results_file)

    logger.info("-"*100)
    catastrophic_forgetting_dict = defaultdict(dict)
    # Traverse over each task in the upstream CL
    for task_num, task_key in enumerate(args.ordered_cl_tasks):

        task_name = task_configs[task_key]['task_name']
        if task_num < 1:
            continue
        logger.info("Evaluating {} model using checkpoint after {} training, on previously-seen tasks {}".format(model_config['encoder_name'],
                                                                                                     task_name,
                                                                                                     ','.join(args.ordered_cl_tasks[:task_num])))
        model_path = os.path.join(output_dir, 'checkpoints', 'task{}_{}'.format(task_num, task_key), 'model')

        # Go from all previous tasks from {0, ..., task_num-1}
        for prev_task_num in range(task_num):

            prev_task_key = args.ordered_cl_tasks[prev_task_num]
            prev_task_config = task_configs[prev_task_key]
            prev_task_name = prev_task_config['task_name']

            # Get evaluation score on prev_task
            prev_task_trainer = task_trainers[prev_task_key]
            eval_score = prev_task_trainer.eval_forgetting(model, model_path)
            logger.info("Evaluation score of {} model on {}, using checkpoint after {} training: {:.2f}".format(args.encoder_name,
                                                                                                  prev_task_name,
                                                                                                  task_name,
                                                                                                  eval_score))
            prev_task_results = cl_results[prev_task_num]
            assert prev_task_results['task_key'] == prev_task_key
            baseline_score = prev_task_results['best_score']
            random_score = task_configs[prev_task_key]['random_baseline_score']

            forgetting_perc = 100.0*(baseline_score - eval_score)/(baseline_score-random_score)
            logger.info("Forgetting of {}, after training on {} = {:.2f}%".format(prev_task_name, task_name, forgetting_perc))

            catastrophic_forgetting_dict[task_key][prev_task_key] = {'prev_task': prev_task_key,
                                                                     'current_task': task_key,
                                                                     'transfer_tasks': '{}->{}'.format(task_num, prev_task_num),
                                                                     'forgetting': forgetting_perc,
                                                                     'absolute_transfer_score': eval_score,
                                                                     'original_prev_task_score': baseline_score,
                                                                     }

    return catastrophic_forgetting_dict