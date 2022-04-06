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
import wandb
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup

from data.visionlanguage_datasets.nlvr2_dataset import build_nlvr2_dataloader

sys.path.insert(0, '.')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["WANDB_START_METHOD"] = "thread"
#wandb.init(project='nlvr')

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

def train_nlvr2(args, model, task_configs, model_config, tokenizer, device, replay_memory=None):

    nlvr_config = task_configs['nlvr2']
    data_dir = nlvr_config['data_dir']
    num_labels = nlvr_config['num_labels']

    # Create model
    batch2inputs_converter = model_config['batch2inputs_converter']
    visual_mode = model_config['visual_mode']
    model.to(device)

    # Create dataloaders for training and validation
    train_dataloader = build_nlvr2_dataloader(args=args,
                                            data_dir=data_dir,
                                            split='train',
                                            visual_mode=visual_mode)

    val_dataloader = build_nlvr2_dataloader(args=args,
                                          data_dir=data_dir,
                                          split='val',
                                          visual_mode=visual_mode)

    # Training hyperparameters
    num_epochs = nlvr_config['num_epochs']
    lr = nlvr_config['lr']
    adam_epsilon = nlvr_config['adam_epsilon']
    weight_decay = nlvr_config['weight_decay']
    warmup_ratio = nlvr_config['warmup_ratio']

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
    max_steps = len(train_dataloader) * num_epochs
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_steps * warmup_ratio),
        num_training_steps=max_steps,
        lr_end=0,
        power=1,
    )

    if args.cl_algorithm == 'experience_replay':
        assert replay_memory is not None
        previous_tasks = list(replay_memory.keys())
        do_replay = True if len(previous_tasks) > 0 else False

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
        for step, batch in enumerate(tqdm(train_dataloader, desc='Training epoch {}'.format(epoch+1))):
            target = batch['labels'].to(device)
            inputs = batch2inputs_converter(batch)

            output = model(task_key='nlvr2', **inputs)
            logits = output[1]
            loss = loss_criterion(logits, target)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                wandb.log({'nlvr': {'loss': loss.item()}})

            if args.cl_algorithm == 'experience_replay' and do_replay is True:
                if (step + 1) % args.replay_frequency == 0:
                    sampled_previous_task = random.choice(previous_tasks)
                    replay_step_method = task_configs[sampled_previous_task]['replay_step_method']
                    replay_loss = replay_step_method(model, replay_memory[sampled_previous_task], task_configs, batch2inputs_converter, device)
                    logger.info("{} replay step: loss = {:.5f}".format(task_configs[sampled_previous_task]['task_name'], replay_loss))

        # Do evaluation after epoch
        eval_score = eval_nlvr2(args, model, val_dataloader, device, batch2inputs_converter)
        logger.info("Evaluation after epoch {}: {:.2f}".format(epoch+1, eval_score))
        wandb.log({'nlvr': {'val_score': eval_score}})

        if eval_score > best_score:
            logger.info("New best evaluation score: {:.2f}".format(eval_score))
            best_score = eval_score
            best_model['epoch'] = epoch
            best_model['model'] = copy.deepcopy(model)

    return best_score, best_model, train_dataloader.dataset

def eval_nlvr2(args, model, val_dataloader, device, batch2inputs_converter):

    model.eval()
    eval_score = 0
    for step, batch in enumerate(tqdm(val_dataloader, desc='Evaluating on NLVR2 val set')):
        inputs = batch2inputs_converter(batch)
        with torch.no_grad():
            output = model(task_key='nlvr2', **inputs)
            logits = output[1]

        batch_scores = (logits.argmax(-1).cpu() == batch['labels'])
        eval_score += batch_scores.sum().item()

    eval_score = eval_score/len(val_dataloader.dataset)*100.0
    logger.info(f'Eval_acc: {eval_score:.3f}')

    model.train()
    return eval_score

def eval_nlvr2_forgetting(args, model, task_configs, model_config, model_path, tokenizer, device):

    nlvr_config = task_configs['nlvr2']
    data_dir = nlvr_config['data_dir']
    num_labels = nlvr_config['num_labels']

    # Create model
    batch2inputs_converter = model_config['batch2inputs_converter']
    visual_mode = model_config['visual_mode']
    model.to(device)

    # Create dataloaders for validation
    val_dataloader = build_nlvr2_dataloader(args=args,
                                          data_dir=data_dir,
                                          split='val',
                                          visual_mode=visual_mode)

    # Load model with encoder weights from encoder_path, and classifier weights from model_path
    model.load_state_dict(torch.load(model_path))

    # Load encoder weights from encoder checkpoint
    #ckpt_encoder_dict = torch.load(encoder_path)
    #model_encoder_dict = model.get_encoder().state_dict()

    #for k in ckpt_encoder_dict.keys():
    #    if model_encoder_dict[k].shape == ckpt_encoder_dict[k].shape:
    #        model_encoder_dict[k].copy_(ckpt_encoder_dict[k])

    return eval_nlvr2(args, model, val_dataloader, device, batch2inputs_converter)

def nlvr2_replay_step(model, nlvr2_replay_memory, task_configs, batch2inputs_converter, device):

    nlvr_config = task_configs['nlvr2']
    # Training hyperparameters
    num_epochs = nlvr_config['num_epochs']
    lr = nlvr_config['lr']
    adam_epsilon = nlvr_config['adam_epsilon']
    weight_decay = nlvr_config['weight_decay']
    warmup_ratio = nlvr_config['warmup_ratio']

    # Create optimizer
    loss_criterion = nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/vilt_utils.py#L236
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon, betas=(0.9, 0.98))
    
    replay_batch = nlvr2_replay_memory.sample_memory_batch()
    target = replay_batch['labels'].to(device)
    inputs = batch2inputs_converter(replay_batch)

    output = model(task_key='nlvr2', **inputs)
    logits = output[1]
    loss = loss_criterion(logits, target)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    wandb.log({'nlvr': {'loss': loss.item()}})

    return loss.item()