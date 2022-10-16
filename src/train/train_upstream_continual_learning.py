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

from cl_algorithms import ExperienceReplayMemory, EWC, AdapterHandler
from cl_algorithms.adapters import ADAPTER_MAP, SUPPORTED_ADAPTER_METHODS
from cl_evaluation.evaluate_cl_algorithm import upstream_knowledge_transfer_eval, catastrophic_forgetting_eval

from configs.model_configs import model_configs, ALLOWED_CL_ENCODERS
from configs.task_configs import task_configs, SUPPORTED_VL_TASKS
from configs.wandb_config import wandb_config

from utils.seed_utils import set_seed
from utils.wandb import wandb_logger

logger = logging.getLogger(__name__)

device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

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
    parser.add_argument("--adapter_method", choices=SUPPORTED_ADAPTER_METHODS,
                        help="Name of Adapter algorithm")
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
    parser.add_argument("--do_wandb_logging", action='store_true',
                        help="Log experiments in W&B.")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for dataloader")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    args = parser.parse_args()
    args.ordered_cl_tasks = args.ordered_cl_tasks.split(',')

    # --------------------- Set up experiment directories ---------------------
    experiment_name = '{}-{}'.format(args.encoder_name, args.cl_algorithm)
    if args.cl_algorithm == 'adapter':
        experiment_name = '{}_{}_{}config'.format(experiment_name, args.adapter_method, args.adapter_config)
    elif args.cl_algorithm == 'freeze_bottom_k_layers':
        experiment_name = experiment_name.replace('_k_layers', '{}layers'.format(args.layers_to_freeze))
    for i, task_key in enumerate(args.ordered_cl_tasks):
        experiment_name = '{}-task{}_{}'.format(experiment_name, i, task_key)
    output_dir = os.path.join(args.output_dir, experiment_name)
    results_file = os.path.join(output_dir, 'results.json')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    set_seed(args)

    # --------------------- Ensure CL algorithm arguments are properly specified  ---------------------
    if args.cl_algorithm == 'singletask_ft':
        assert len(args.ordered_cl_tasks) == 1
    else:
        assert len(args.ordered_cl_tasks) > 1
    if args.cl_algorithm == 'experience_replay':
        assert args.memory_percentage > 0.0
        assert args.replay_frequency > 0
    if args.cl_algorithm == 'adapter':
        assert args.adapter_reduction_factor > 0
    if args.cl_algorithm == 'ewc':
        assert args.ewc_fisher_sample_percentage > 0
        assert args.ewc_loss_weight > 0.0
    if args.cl_algorithm == 'freeze_bottom_k_layers':
        assert args.layers_to_freeze > 0


    # --------------------- Ensure all the tasks for continual learning are supported VL tasks  ---------------------
    for task_key in args.ordered_cl_tasks:
        assert task_key in SUPPORTED_VL_TASKS

    # --------------------- Load the ContinualLearner model, based on encoder_name argument  ---------------------
    model_config = model_configs[args.encoder_name]
    create_model_method = create_continual_learner_map[args.encoder_name]
    model = create_model_method(model_name_or_path=args.pretrained_model_name,
                                ordered_cl_tasks=args.ordered_cl_tasks, 
                                model_config=model_config, 
                                task_configs=task_configs,
                                device=device)
    args.visual_input_type = model_config['visual_input_type']


    # --------------------- CL algorithm-specific initializations  ------------------------------------------
    # No specific initializations for single-task finetuning and sequential finetuning

    replay_memory = None
    ewc = None
    adapter_handler = None
    if args.cl_algorithm == 'experience_replay':
        # Initialize an empty replay memory
        replay_memory = ExperienceReplayMemory()

    elif args.cl_algorithm == 'adapter':
        # Create AdapterHandler, and add Adapters for each task
        adapter_handler = AdapterHandler(adapter_method=args.adapter_method, args=args)
        adapter_handler.add_adapters_to_model(model)

    elif args. cl_algorithm == 'ewc':
        ewc = EWC(args)

    elif args.cl_algorithm == 'freeze_encoder':
        # Freeze encoder weights
        model.get_encoder().freeze_all_weights()

    elif args.cl_algorithm == 'freeze_bottom_k_layers':
        # Freeze bottom K layers
        model.get_encoder().freeze_bottom_k_layers(k=args.layers_to_freeze)

    # ------------------------------------------ Print some model info ------------------------------------------
    logger.info("Succesfully initialized {}-based Continual Learner".format(model_config['encoder_name']))
    logger.info("{} task heads: {}".format(len(args.ordered_cl_tasks), ','.join(args.ordered_cl_tasks)))
    logger.info("CL Algorithm: {}".format(args.cl_algorithm))
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Total Parameters: {:.2f}M'.format(total_params*10**-6))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    logger.info('Trainable Parameters: {:.2f}M ({:.2f}%)'.format(trainable_params*10**-6, (trainable_params/total_params*100)))
    logger.info('Model checkpoints saved to {}'.format(output_dir))
    logger.info("-"*100)

    if args.do_train:

        # Create W&B experiment
        if args.do_wandb_logging:
            logger.info('W&B project: {}, experiment: {}'.format(wandb_config['project_name'], experiment_name))
            wandb_logger.initialize(wandb_config=wandb_config,
                                    experiment_name=experiment_name)

        # If continuing an already-started experiment, and old upstream results were detected, then load the old results from json
        results = []
        if os.path.exists(results_file):
            results = json.load(open(results_file))
            logger.info("-"*100)
            logger.info("Cached results:")
            for i, r in enumerate(results):
                task_key = r['task_key']
                best_score = r['best_score']
                logger.info("Task #{}: {} - best score = {:.2f}".format(i+1, task_configs[task_key]['task_name'], best_score))
        task_trainers = {}

        # Begin training on VL tasks sequentially
        logger.info("-"*100)
        logger.info("Training models on Vision-Language continual learning tasks...")
        for task_num, task_key in enumerate(args.ordered_cl_tasks):

            logger.info("-"*100)
            task_name = task_configs[task_key]['task_name']
            task_output_dir = os.path.join(output_dir, 'checkpoints', 'task{}_{}'.format(task_num, task_key))

            if os.path.exists(os.path.join(task_output_dir, 'model')):

                # If we find model checkpoint for this task, load the checkpoint and move onto next CL task
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
                logger.info("Loaded model checkpoint from task {}! Moving on to next task...".format(task_name))

                task_trainer_class = task_configs[task_key]['task_trainer']
                task_trainer = task_trainer_class(args, task_configs, model_config, device)

            else:

                #If CL algorithm is adapters, activate adapter for this task
                if args.cl_algorithm == 'adapter':
                    logger.info("Activating adapter networks only for task {}".format(task_name))
                    adapter_handler.activate_adapter_for_training(task_key=task_key, model=model)
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
                    logger.info('Trainable Parameters: {:.2f}M ({:.2f}%)'.format(trainable_params*10**-6, (trainable_params/total_params*100)))

                # Create the Trainer method for the current CL task, and call the train method
                logger.info("Training {} model on task #{}: {}".format(args.encoder_name, task_num+1, task_name))
                task_trainer_class = task_configs[task_key]['task_trainer']
                task_trainer = task_trainer_class(args, task_configs, model_config, device)
                best_eval_score, best_model = task_trainer.train(model,
                                                                replay_memory=replay_memory,
                                                                ewc=ewc)
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

            task_trainers[task_key] = task_trainer
            if args.cl_algorithm == 'experience_replay':
                # If doing experience replay, create memory buffer for current task
                replay_memory.add_task_memory_buffer(args=args,
                                                     task_key=task_key,
                                                     task_config=task_configs[task_key],
                                                     task_trainer=task_trainer,
                                                     memory_percentage=args.memory_percentage,
                                                     sampling_strategy=args.memory_sampling_strategy)
            elif args.cl_algorithm == 'ewc' and task_num < len(args.ordered_cl_tasks)-1:
                # If doing EWC, save task parameters and compute Fisher information matrix over the training set
                ewc.save_task_parameters(task_key=task_key,
                                        model=model,
                                        task_trainer=task_trainer,
                                        device=device)

    if args.do_eval:

        logger.info("-"*100)

        # Forward transfer from continual learning, by comparing to single-task finetuning score
        logger.info("Evaluating FORWARD TRANSFER of {} model on {}".format(args.encoder_name, ' -> '.join(args.ordered_cl_tasks)))
        upstream_knowledge_dict = upstream_knowledge_transfer_eval(args, results_file)
        average_relative_gain = sum(list([x['relative_gain'] for x in upstream_knowledge_dict.values()]))/len(upstream_knowledge_dict)
        logger.info("Average forward transfer gain = {:.2f}%".format(average_relative_gain))

        logger.info("-"*100)

        # Forgetting evaluation
        if not args.do_train:
            logger.info("Creating task trainers for forgetting evaluation...")
            task_trainers = {}
            for task_num, task_key in enumerate(args.ordered_cl_tasks):
                task_trainer_class = task_configs[task_key]['task_trainer']
                task_trainer = task_trainer_class(args, task_configs, model_config, device)
                task_trainers[task_key] = task_trainer
        else:
            for task_num, task_key in enumerate(args.ordered_cl_tasks):
                assert task_key in task_trainers.keys()

        # Run the forgetting evaluation, and save results to file
        logger.info("Evaluating CATASTROPHIC FORGETTING of {} model on {}".format(args.encoder_name, ' -> '.join(args.ordered_cl_tasks)))
        catastrophic_forgetting_dict = catastrophic_forgetting_eval(args, results_file, model, task_trainers, adapter_handler)

        eval_results_file = os.path.join(output_dir, 'eval_results.json')
        eval_results = {'upstream_knowledge_transfer': upstream_knowledge_dict,
                        'forgetting': catastrophic_forgetting_dict}
        json.dump(eval_results, open(eval_results_file, 'w'))

        logger.info("-"*100)

if __name__ == '__main__':
    main()
