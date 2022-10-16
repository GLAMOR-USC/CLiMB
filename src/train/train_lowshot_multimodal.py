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

sys.path.insert(0, '.')

import numpy as np
import torch
from tqdm import tqdm

from transformers.adapters import AdapterConfig

from modeling import load_encoder_map, create_continual_learner_map

from cl_algorithms import ExperienceReplayMemory, EWC
from cl_evaluation.evaluate_cl_algorithm import upstream_knowledge_transfer_eval, catastrophic_forgetting_eval
from configs.model_configs import model_configs, ALLOWED_CL_ENCODERS
from configs.task_configs import task_configs, SUPPORTED_VL_TASKS
from configs.adapter_configs import ADAPTER_MAP
from utils.seed_utils import set_seed

logger = logging.getLogger(__name__)

device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def train_low_shot(args, low_shot_model, low_shot_task_key, model_config, device):

    low_shot_task_name = task_configs[low_shot_task_key]['task_name']
    low_shot_config = task_configs[low_shot_task_key]['low_shot_config']

    # Create the Trainer method for the current CL task, and call the train method
    logger.info("Training {} model on low-shot task {}, low_shot_config={}".format(args.encoder_name,
                                                                             low_shot_task_name, 
                                                                             low_shot_config))
    task_trainer_class = low_shot_config['task_trainer']
    task_trainer = task_trainer_class(args, task_configs, model_config, device, low_shot_config=low_shot_config)
    best_eval_score, best_model = task_trainer.train(low_shot_model)

    return best_eval_score, low_shot_config

def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--encoder_name", default=None, type=str, required=True, choices=ALLOWED_CL_ENCODERS,
                        help="The name of the base pretrained encoder.")
    parser.add_argument("--pretrained_model_name", default=None, type=str, required=True,
                        help="Name of pretrained model weights to load.")
    parser.add_argument("--ordered_cl_tasks", type=str, required=True,
                        help="Ordered list of VL task keys for continual learning, seprated by commas.")
    parser.add_argument("--cl_algorithm", type=str, required=True, choices=['singletask_ft',
                                                                            'sequential_ft',
                                                                            'experience_replay',
                                                                            'ewc',
                                                                            'adapter',
                                                                            'freeze_encoder',
                                                                            'freeze_bottom_k_layers'],
                        help="Name of Continual Learning algorithm used.")
    parser.add_argument("--climb_data_dir", type=str, required=True, default='/data/datasets/MCL/',
                        help="Directory where all the MCL data is stored")

    # Arguments specific to experience replay algorithm
    parser.add_argument("--memory_percentage", type=float, default=0.0,
                        help="Percentage of tasks' training samples saved into memory.")
    parser.add_argument("--memory_sampling_strategy", type=str, choices=['random', 'random-balanced'],
                        help="Strategy for sampling memory buffer samples.")
    parser.add_argument("--replay_frequency", type=int,
                        help="Number of training steps after which to do a memory replay step.")

    # Arguments specific to Adapters algorithm
    parser.add_argument("--adapter_config", choices=list(ADAPTER_MAP.keys()),
                        help="Type of Adapter architecture")
    parser.add_argument("--adapter_reduction_factor", type=int, default=0,
                        help="Downsampling ratio for adapter layers")

    # Arguments specific to EWC algorithm
    parser.add_argument("--ewc_fisher_sample_percentage", type=float, default=0.0,
                        help="Percentage of training samples for computing Fisher information matrix per task")
    parser.add_argument("--ewc_loss_weight", type=float, default=0.0,
                        help="Factoring for scaling the EWC loss")

    # Arguments specific to frozen bottom-k layers algorithm
    parser.add_argument("--layers_to_freeze", type=int, default=0,
                        help="Number of layers to freeze (if freezing bottom-k layers)")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Name of output directory, where all experiment results and checkpoints are saved.")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for dataloader")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    args = parser.parse_args()
    args.ordered_cl_tasks = args.ordered_cl_tasks.split(',')

    # --------------------- Set up experiment directories
    experiment_name = '{}-{}'.format(args.encoder_name, args.cl_algorithm)
    if args.cl_algorithm == 'adapter':
        experiment_name = '{}_{}'.format(experiment_name, args.adapter_config)
    elif args.cl_algorithm == 'freeze_bottom_k_layers':
        experiment_name = experiment_name.replace('_k_layers', '{}layers'.format(args.layers_to_freeze))
    for i, task_key in enumerate(args.ordered_cl_tasks):
        experiment_name = '{}-task{}_{}'.format(experiment_name, i, task_key)
    output_dir = os.path.join(args.output_dir, experiment_name)
    results_file = os.path.join(output_dir, 'lowshot_results.json')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    set_seed(args)

    # --------------------- Ensure all the tasks for continual learning are supported VL tasks  ---------------------
    for task_key in args.ordered_cl_tasks:
        assert task_key in SUPPORTED_VL_TASKS

    # --------------------- Load the correct ContinualLeaner model, based on encoder_name argument  ---------------------
    model_config = model_configs[args.encoder_name]
    create_model_method = create_continual_learner_map[args.encoder_name]
    model = create_model_method(model_name_or_path=args.pretrained_model_name,
                                ordered_cl_tasks=args.ordered_cl_tasks, 
                                model_config=model_config, 
                                task_configs=task_configs,
                                device=device)
    args.visual_input_type = model_config['visual_input_type']

    # ------------------------------------------ Print some model info ------------------------------------------
    logger.info("Succesfully initialized {}-based Continual Learner".format(model_config['encoder_name']))
    logger.info("{} task heads: {}".format(len(args.ordered_cl_tasks), ','.join(args.ordered_cl_tasks)))
    logger.info("CL Algorithm: {}".format(args.cl_algorithm))
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Total Parameters: {:.2f}M'.format(total_params*10**-6))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    logger.info('Trainable Parameters: {:.2f}M ({:.2f}%)'.format(trainable_params*10**-6, (trainable_params/total_params*100)))
    logger.info('Model checkpoints loaded from {}'.format(output_dir))
    logger.info("-"*100)

    results = []
    if os.path.exists(results_file):
        results = json.load(open(results_file))
        logger.info("-"*100)
        logger.info("Cached results:")
        for i, r in enumerate(results):
            task_key = r['task_key']
            best_score = r['best_low_shot_score']
            logger.info("Task #{}: {} - best score = {:.2f}".format(i+1, task_configs[task_key]['task_name'], best_score))
    task_trainers = {}

    logger.info("-"*100)
    logger.info("Doing low-shot transfer to Vision-Language tasks...")

    if args.cl_algorithm == 'singletask_ft':
        # --------------------- Do low-shot training of pre-trained encoder on a single task --------------------- 
        task_key = args.ordered_cl_tasks[0]
        low_shot_model = copy.deepcopy(model)
        low_shot_eval_score, low_shot_config = train_low_shot(args, low_shot_model, task_key, model_config, device)
        logger.info("Best {} evaluation score = {:.2f}".format(task_key, low_shot_eval_score))

        # --------------------- Save low-shot results ---------------------
        config_copy = copy.deepcopy(low_shot_config)
        config_copy.pop('task_trainer', None)
        task_results = {
            'task_key': task_key,
            'best_low_shot_score': low_shot_eval_score,
            'low_shot_config': config_copy,
        }
        results.append(task_results)
        json.dump(results, open(results_file, 'w'))
        logger.info("Saved low-shot transfer results so far!")

    else:
        # Iterate through each task, load its checkpoint and do low-shot training on all tasks after it
        for task_num, task_key in enumerate(args.ordered_cl_tasks):

            #  --------------------- Find model checkpoint for this task, load the checkpoint and move onto next CL task ---------------------
            logger.info("-"*100)
            task_name = task_configs[task_key]['task_name']
            task_output_dir = os.path.join(output_dir, 'checkpoints', 'task{}_{}'.format(task_num, task_key))

            assert os.path.exists(os.path.join(task_output_dir, 'model'))
            logger.info("Found checkpoint for task {}!".format(task_name))
            try:
                model.load_state_dict(torch.load(os.path.join(task_output_dir, 'model')))
            except Exception as e:
                ckpt_state_dict = torch.load(os.path.join(task_output_dir, 'model'))
                initialized = {k: False for k in model.state_dict().keys()}
                for k in ckpt_state_dict.keys():
                    model.state_dict()[k].copy_(ckpt_state_dict[k])
                    initialized[k] = True
                logger.info("Uninitialized keys: {}".format(','.join([k for k in initialized.keys() if initialized[k] is False])))
                torch.save(model.state_dict(), os.path.join(task_output_dir, 'model'))
                logger.info("Saved model with uninitialized keys as new checkpoint")
            logger.info("Loaded model checkpoint from task {}!".format(task_name))

            # --------------------- Do low-shot training on all tasks after task_num --------------------- 
            low_shot_tasks = args.ordered_cl_tasks[task_num+1:]
            logger.info("Doing low-shot transfer to tasks {} using checkpoint from {}".format(','.join(low_shot_tasks), task_name))

            for low_shot_task_key in low_shot_tasks:
                low_shot_task_num = args.ordered_cl_tasks.index(low_shot_task_key)
                low_shot_model = copy.deepcopy(model)
                low_shot_eval_score, low_shot_config = train_low_shot(args, low_shot_model, low_shot_task_key, model_config, device)
                logger.info("Best {} evaluation score = {:.2f}".format(low_shot_task_key, low_shot_eval_score))


                # --------------------- Save low-shot results so far ---------------------
                config_copy = copy.deepcopy(low_shot_config)
                config_copy.pop('task_trainer', None)
                task_results = {
                    'upstream_task_num': task_num,
                    'upstream_task_key': task_key,
                    'lowshot_task_num': low_shot_task_num,
                    'lowshot_task_key': low_shot_task_key,
                    'best_low_shot_score': low_shot_eval_score,
                    'low_shot_config': config_copy,
                }
                results.append(task_results)
                json.dump(results, open(results_file, 'w'))
                logger.info("Saved low-shot transfer results so far!")
                logger.info("-"*100)
            logger.info("-"*100)

if __name__ == '__main__':
    main()
