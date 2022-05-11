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
import pdb
from tqdm import tqdm
import wandb

sys.path.insert(0, '.')

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup

from data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset
from data.visionlanguage_datasets.vcr_dataset import build_vcr_dataloader

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

def get_vcr_train_dataset(args, task_configs, model_config, tokenizer):

    vcr_config = task_configs['vcr']
    data_dir = os.path.join(args.mcl_data_dir, vcr_config['data_dir'])
    task_type = vcr_config['task_type']
    visual_mode = model_config['visual_mode']

    # Create dataloaders for training and validation
    vcr_train_dataloader = build_vcr_dataloader(args=args,
                                                data_dir=data_dir,
                                                split='train',
                                                tokenizer=tokenizer,
                                                task_type=task_type,
                                                visual_mode=visual_mode)
    return vcr_train_dataloader.dataset

def train_vcr(args, model, task_configs, model_config, tokenizer, device, replay_memory=None):

    vcr_config = task_configs['vcr']
    data_dir = os.path.join(args.mcl_data_dir, vcr_config['data_dir'])
    num_labels = vcr_config['num_labels']
    task_type = vcr_config['task_type']

    # Create model
    visual_mode = model_config['visual_mode']
    batch2inputs_converter = model_config['batch2inputs_converter']
    model.to(device)
    if args.cl_algorithm == 'adapter':
        model.set_active_adapters("vcr")

    # Create dataloaders for training and validation
    vcr_train_dataloader = build_vcr_dataloader(args=args,
                                                data_dir=data_dir,
                                                split='train',
                                                tokenizer=tokenizer,
                                                task_type=task_type,
                                                visual_mode=visual_mode)

    vcr_val_dataloader = build_vcr_dataloader(args=args,
                                              data_dir=data_dir,
                                              split='val',
                                              tokenizer=tokenizer,
                                              task_type=task_type,
                                              visual_mode=visual_mode)

    # Training hyperparameters
    num_epochs = vcr_config['num_epochs']
    lr = vcr_config['lr']
    adam_epsilon = vcr_config['adam_epsilon']
    weight_decay = vcr_config['weight_decay']

    # Create optimizer
    loss_criterion = nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/vilt_utils.py#L236
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon, betas=(0.9, 0.98))
    # Create Scheduler
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/vilt_utils.py#L263
    max_steps = len(vcr_train_dataloader) * num_epochs
    warmup_ratio = 0.1 # TODO remove hard code
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_steps * warmup_ratio),
        num_training_steps=max_steps,
        lr_end=0,
        power=1,
    )

    if args.cl_algorithm == 'experience_replay':
        assert replay_memory is not None
        do_replay = replay_memory.do_replay()

    best_score = 0
    best_model = {
        'epoch': 0,
        'model': copy.deepcopy(model), #model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }

    model.zero_grad()
    model.train()
    for epoch in range(num_epochs):
        # Training loop for epoch
        for step, batch in enumerate(tqdm(vcr_train_dataloader, desc='Training epoch {}'.format(epoch+1))):
            # Convert inputs into expected format
            inputs = batch2inputs_converter(batch)
            target = batch['labels'].to(device)

            # Forward pass
            output = model(task_key='vcr', **inputs)
            logits = output[1]
            loss = loss_criterion(logits, target)

            # Back propogate
            loss.backward()

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                wandb.log({'vcr': {'loss': loss.item()}})

            if args.cl_algorithm == 'experience_replay' and do_replay is True:
                if (step + 1) % args.replay_frequency == 0:
                    sampled_replay_task = replay_memory.sample_replay_task()
                    replay_args = {'model': model,
                                   'task_configs': task_configs,
                                   'batch2inputs_converter': batch2inputs_converter,
                                   'device': device}
                    replay_loss = replay_memory.run_replay_step(sampled_replay_task, **replay_args)
                    logger.info("{} replay step: loss = {:.5f}".format(task_configs[sampled_replay_task]['task_name'], replay_loss))

        # Do evaluation after epoch
        eval_score = eval_vcr(args, model, vcr_val_dataloader, device, batch2inputs_converter)
        logger.info("Evaluation after epoch {}: {:.2f}".format(epoch+1, eval_score))
        wandb.log({'vcr': {'val_score': eval_score}})
        if eval_score > best_score:
            logger.info("New best evaluation score: {:.2f}".format(eval_score))
            best_score = eval_score
            best_model['epoch'] = epoch
            best_model['model'] = copy.deepcopy(model)

    return best_score, best_model, vcr_train_dataloader.dataset

def eval_vcr(args, model, vcr_val_dataloader, device, batch2inputs_converter):

    model.eval()
    eval_correct = 0

    for step, batch in enumerate(tqdm(vcr_val_dataloader, desc='Evaluating on VQA val set')):
        inputs = batch2inputs_converter(batch)
        target = batch['labels'].to(device)

        #output = model(images=images, texts=texts)      # TODO: Create abstraction that can convert batch keys into model input keys for all models
        with torch.no_grad():
            output = model(task_key='vcr', **inputs)
        logits = output[1]

        batch_scores = (logits.argmax(-1).cpu() == batch['labels'])
        eval_correct += batch_scores.sum().item()

    eval_acc = eval_correct/len(vcr_val_dataloader.dataset)*100.0

    model.train()
    return eval_acc

def eval_vcr_forgetting(args, model, model_path, task_configs, model_config, tokenizer, device):

    vcr_config = task_configs['vcr']
    data_dir = os.path.join(args.mcl_data_dir, vcr_config['data_dir'])
    num_labels = vcr_config['num_labels']
    task_type = vcr_config['task_type']

    visual_mode = model_config['visual_mode']
    batch2inputs_converter = model_config['batch2inputs_converter']
    model.to(device)
    if args.cl_algorithm == 'adapter':
        model.set_active_adapters("vcr")

    vcr_val_dataloader = build_vcr_dataloader(args=args,
                                              data_dir=data_dir,
                                              split='val',
                                              tokenizer=tokenizer,
                                              task_type=task_type,
                                              visual_mode=visual_mode)

    # Load model with encoder weights from encoder_path, and classifier weights from model_path
    model.load_state_dict(torch.load(model_path))
    logger.info("Loaded model checkpoint from {}".format(model_path))

    # Load encoder weights from encoder checkpoint
    #ckpt_encoder_dict = torch.load(encoder_path)
    #model_encoder_dict = model.get_encoder().state_dict()

    #for k in ckpt_encoder_dict.keys():
    #    if model_encoder_dict[k].shape == ckpt_encoder_dict[k].shape:
    #        model_encoder_dict[k].copy_(ckpt_encoder_dict[k])

    return eval_vcr(args, model, vcr_val_dataloader, device, batch2inputs_converter)
