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
import wandb

from transformers import BertTokenizer
from transformers.adapters import AdapterConfig

from modeling import load_encoder_map, continual_learner_map

from cl_algorithms import ExperienceReplayMemory
from cl_evaluation.evaluate_cl_algorithm import forward_transfer_eval, catastrophic_forgetting_eval
from configs.model_configs import model_configs
from configs.task_configs import task_configs, SUPPORTED_VL_TASKS
from configs.adapter_configs import ADAPTER_MAP
from utils.seed_utils import set_seed

logger = logging.getLogger(__name__)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--encoder_name", default=None, type=str, required=True, choices=['vilt'],
                        help="The name of the base pretrained encoder.")
    parser.add_argument("--pretrained_model_name", default=None, type=str, required=True,
                        help="Name of pretrained model weights to load.")
    parser.add_argument("--ordered_cl_tasks", type=str, required=True,
                        help="Ordered list of VL task keys for continual learning, seprated by commas.")
    parser.add_argument("--cl_algorithm", type=str, required=True, choices=['singletask_ft', 'sequential_ft', 'experience_replay', 'adapter'],
                        help="Name of Continual Learning algorithm used.")
    parser.add_argument("--mcl_data_dir", type=str, required=True, default='/data/datasets/MCL/',
                        help="Directory where all the MCL data is stored")
    parser.add_argument("--do_train", action='store_true',
                        help="If True, train the model on these tasks")
    parser.add_argument("--do_eval", action='store_true',
                        help="If True, evaluate the model on these tasks.")

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

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Name of output directory, where all experiment results and checkpoints are saved.")
    parser.add_argument("--wandb_project_name", type=str, default="climb-cl",
                        help="Name of W&B project where experiments are logged.")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for dataloader")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    args = parser.parse_args()
    args.ordered_cl_tasks = args.ordered_cl_tasks.split(',')

    # Set up experiment directories
    experiment_name = '{}-{}'.format(args.encoder_name, args.cl_algorithm)
    if args.cl_algorithm == 'adapter':
        experiment_name = '{}_{}'.format(experiment_name, args.adapter_config)
    for i, task_key in enumerate(args.ordered_cl_tasks):
        experiment_name = '{}-task{}_{}'.format(experiment_name, i, task_key)
    output_dir = os.path.join(args.output_dir, experiment_name)
    results_file = os.path.join(output_dir, 'results.json')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    set_seed(args)

    # Ensure CL algorithm arguments are properly specified
    if args.cl_algorithm == 'singletask_ft':
        assert len(args.ordered_cl_tasks) == 1
    else:
        assert len(args.ordered_cl_tasks) > 1
    if args.cl_algorithm == 'experience_replay':
        assert args.memory_percentage > 0.0
        assert args.replay_frequency > 0


    # Ensure all the tasks for continual learning are supported VL tasks
    for task_key in args.ordered_cl_tasks:
        assert task_key in SUPPORTED_VL_TASKS

    # Load the correct Encoder model, based on encoder_name argument
    model_config = model_configs[args.encoder_name]
    load_encoder_method = load_encoder_map[args.encoder_name]
    encoder = load_encoder_method(args.pretrained_model_name, device)
    continual_learner_class = continual_learner_map[args.encoder_name]
    model = continual_learner_class(args.ordered_cl_tasks, encoder, model_config['encoder_dim'], task_configs)
    args.visual_mode = model_config['visual_mode']

    # Add Adapters for each task
    if args.cl_algorithm == 'adapter':
        adapter_config = AdapterConfig.load(args.adapter_config)
        config_dict = adapter_config.to_dict()
        if args.adapter_reduction_factor > 0:
            config_dict['reduction_factor'] = args.adapter_reduction_factor
        adapter_config = AdapterConfig.from_dict(config_dict)
        logger.info("Adding Adapter layers with configuration:")
        logger.info(str(adapter_config))
        for task_key in args.ordered_cl_tasks:
            model.add_adapter(task_key, config=adapter_config)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Total Parameters: {:.2f}M'.format(total_params*10**-6))

    if args.do_train:

        #assert len(os.listdir(output_dir)) == 0         # Ensure I am not overwriting an existing output directory

        # Create W&B experiment
        logger.info('W&B project: {}, experiment: {}'.format(args.wandb_project_name, experiment_name))
        wandb.init(project=args.wandb_project_name,
            name=experiment_name,
            entity='las-cl',
            reinit=True)

        results = []
        if args.cl_algorithm == 'experience_replay':
            replay_memory = ExperienceReplayMemory()
        else:
            replay_memory = None

        logger.info("-"*100)
        logger.info("Training models on Vision-Language continual learning tasks...")
        for task_num, task_key in enumerate(args.ordered_cl_tasks):

            task_name = task_configs[task_key]['task_name']
            task_output_dir = os.path.join(output_dir, 'checkpoints', 'task{}_{}'.format(task_num, task_key))

            if os.path.isfile(os.path.join(task_output_dir, 'model')):
                logger.info("Found checkpoint for task {}!".format(task_name))
                model.load_state_dict(torch.load(os.path.join(task_output_dir, 'model')))
                logger.info("Loaded model checkpoint from task {}! Moving on to next task...".format(task_name))

                get_train_dataset_method = task_configs[task_key]['get_train_dataset_method']
                task_train_dataset = get_train_dataset_method(args, task_configs, model_config, tokenizer)

            else:

                # Load the correct training method for current CL task, and call the training method
                logger.info("-"*100)

                if args.cl_algorithm == 'adapter':
                    model.train_adapter(task_key)
                    model.set_active_adapters(task_key)
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
                    logger.info('Trainable Parameters: {:.2f}M ({:.2f}%)'.format(trainable_params*10**-6, (trainable_params/total_params*100)))

                logger.info("Training {} model on task #{}: {}".format(args.encoder_name, task_num+1, task_name))
                train_method = task_configs[task_key]['train_method']
                best_eval_score, best_model, task_train_dataset = train_method(args, model, task_configs, model_config, tokenizer, device, replay_memory)

                logger.info("Best {} evaluation score = {:.2f}, after epoch {}".format(task_name, best_eval_score, best_model['epoch']+1))

                # Save best model checkpoint, and separately save the models' Encoder object
                logger.info("Saving best model and encoder checkpoint after {} training".format(task_name))
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

            if args.cl_algorithm == 'experience_replay':
                replay_memory.add_task_memory_buffer(args=args,
                                                     task_key=task_key,
                                                     task_config=task_configs[task_key],
                                                     train_dataset=task_train_dataset,
                                                     memory_percentage=args.memory_percentage,
                                                     sampling_strategy=args.memory_sampling_strategy)

    if args.do_eval:

        logger.info("-"*100)
        logger.info("Evaluating FORWARD TRANSFER of {} model on {}".format(args.encoder_name, ' -> '.join(args.ordered_cl_tasks)))
        forward_transfer_dict = forward_transfer_eval(args, results_file)
        average_relative_gain = sum(list(forward_transfer_dict.values()))/len(forward_transfer_dict)
        logger.info("Average forward transfer gain = {:.2f}%".format(average_relative_gain))
        logger.info("-"*100)

        logger.info("Evaluating CATASTROPHIC FORGETTING of {} model on {}".format(args.encoder_name, ' -> '.join(args.ordered_cl_tasks)))
        catastrophic_forgetting_dict = catastrophic_forgetting_eval(args, results_file, model, tokenizer, device)
        # TODO: Aggregate catastrophic forgetting results
        logger.info("-"*100)

if __name__ == '__main__':
    main()
