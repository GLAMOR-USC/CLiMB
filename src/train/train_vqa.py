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
from data.visionlanguage_datasets.vqa_dataset import build_vqa_dataloader

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

def compute_score_with_logits(logits, labels, device):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def train_vqa(args, model, task_configs, model_config, tokenizer, device, memory_buffers=None):

    vqa_config = task_configs['vqa']
    data_dir = vqa_config['data_dir']
    num_labels = vqa_config['num_labels']

    # Load COCO Images dataset for image data backbone
    images_source = vqa_config['images_source']
    mscoco_config = task_configs[images_source]
    images_dataset = MSCOCOImagesDataset(mscoco_config['data_dir'])

    # Create model
    visual_mode = model_config['visual_mode']
    batch2inputs_converter = model_config['batch2inputs_converter']
    model.to(device)

    # Create dataloaders for training and validation
    vqa_train_dataloader = build_vqa_dataloader(args=args,
                                                data_dir=data_dir,
                                                images_dataset=images_dataset,
                                                split='train',
                                                tokenizer=tokenizer,
                                                visual_mode=visual_mode)

    vqa_val_dataloader = build_vqa_dataloader(args=args,
                                              data_dir=data_dir,
                                              images_dataset=images_dataset,
                                              split='val',
                                              tokenizer=tokenizer,
                                              visual_mode=visual_mode)

    # Training hyperparameters
    num_epochs = vqa_config['num_epochs']
    lr = vqa_config['lr']
    adam_epsilon = vqa_config['adam_epsilon']
    weight_decay = vqa_config['weight_decay']

    # Create optimizer
    loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/vilt_utils.py#L236
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon, betas=(0.9, 0.98))
    # Create Scheduler
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/vilt_utils.py#L263
    max_steps = len(vqa_train_dataloader) * num_epochs
    warmup_ratio = 0.1 # TODO remove hard code
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_steps * warmup_ratio),
        num_training_steps=max_steps,
        lr_end=0,
        power=1,
    )

    if args.cl_algorithm == 'experience_replay':
        assert memory_buffers is not None
        previous_tasks = list(memory_buffers.keys())
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
        for step, batch in enumerate(tqdm(vqa_train_dataloader, desc='Training epoch {}'.format(epoch+1))):
            inputs = batch2inputs_converter(batch)
            target = batch['target_scores'].to(device)

            #output = model(images=images, texts=texts)      # TODO: Create abstraction that can convert batch keys into model input keys for all models
            output = model(task_key='vqa', **inputs)
            logits = output[1]
            # https://github.com/dandelin/ViLT/blob/master/vilt/modules/objectives.py#L317
            loss = loss_criterion(logits, target) * target.shape[1]

            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                wandb.log({'vqa': {'loss': loss.item()}})

            if args.cl_algorithm == 'experience_replay' and do_replay is True:
                if (step + 1) % args.replay_frequency == 0:
                    sampled_previous_task = random.choice(previous_tasks)
                    replay_step_method = task_configs[sampled_previous_task]['replay_step_method']
                    replay_loss = replay_step_method(model, memory_buffers[sampled_previous_task], task_configs, batch2inputs_converter, device)
                    logger.info("{} replay step: loss = {:.5f}".format(task_configs[sampled_previous_task]['task_name'], replay_loss))

        # Do evaluation after epoch
        eval_score = eval_vqa(args, model, vqa_val_dataloader, device, batch2inputs_converter)
        logger.info("Evaluation after epoch {}: {:.2f}".format(epoch+1, eval_score))
        wandb.log({'vqa': {'val_score': eval_score}})
        if eval_score > best_score:
            logger.info("New best evaluation score: {:.2f}".format(eval_score))
            best_score = eval_score
            best_model['epoch'] = epoch
            best_model['model'] = copy.deepcopy(model)

    return best_score, best_model, vqa_train_dataloader.dataset

def eval_vqa(args, model, vqa_val_dataloader, device, batch2inputs_converter):

    model.eval()
    eval_score = 0

    for step, batch in enumerate(tqdm(vqa_val_dataloader, desc='Evaluating on VQA val set')):
        inputs = batch2inputs_converter(batch)
        target = batch['target_scores'].to(device)

        #output = model(images=images, texts=texts)      # TODO: Create abstraction that can convert batch keys into model input keys for all models
        with torch.no_grad():
            output = model(task_key='vqa', **inputs)
        logits = output[1]

        answer_scores = compute_score_with_logits(logits, target, device)
        batch_scores = torch.sum(answer_scores, 1)

        eval_score += batch_scores.sum().item()

    eval_score = eval_score/len(vqa_val_dataloader.dataset)*100.0

    model.train()
    return eval_score

def eval_vqa_forgetting(args, model, task_configs, model_config, model_path, tokenizer, device):

    vqa_config = task_configs['vqa']
    data_dir = vqa_config['data_dir']
    num_labels = vqa_config['num_labels']

    images_source = vqa_config['images_source']
    mscoco_config = task_configs[images_source]
    images_dataset = MSCOCOImagesDataset(mscoco_config['data_dir'])

    visual_mode = model_config['visual_mode']
    batch2inputs_converter = model_config['batch2inputs_converter']
    model.to(device)

    vqa_val_dataloader = build_vqa_dataloader(args=args,
                                          data_dir=data_dir,
                                          images_dataset=images_dataset,
                                          split='val',
                                          tokenizer=tokenizer,
                                          visual_mode=visual_mode)

    # Load model with encoder weights from encoder_path, and classifier weights from model_path
    model.load_state_dict(torch.load(model_path))

    # Load encoder weights from encoder checkpoint
    #ckpt_encoder_dict = torch.load(encoder_path)
    #model_encoder_dict = model.get_encoder().state_dict()

    #for k in ckpt_encoder_dict.keys():
    #    if model_encoder_dict[k].shape == ckpt_encoder_dict[k].shape:
    #        model_encoder_dict[k].copy_(ckpt_encoder_dict[k])

    return eval_vqa(args, model, vqa_val_dataloader, device, batch2inputs_converter)

def vqa_replay_step(model, vqa_replay_memory, task_configs, batch2inputs_converter, device):

    vqa_config = task_configs['vqa']
    # Training hyperparameters
    num_epochs = vqa_config['num_epochs']
    lr = vqa_config['lr']
    adam_epsilon = vqa_config['adam_epsilon']
    weight_decay = vqa_config['weight_decay']

    # Create optimizer
    loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/vilt_utils.py#L236
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon, betas=(0.9, 0.98))
    
    replay_batch = vqa_replay_memory.sample_memory_batch()
    inputs = batch2inputs_converter(replay_batch)
    target = replay_batch['target_scores'].to(device)

    #output = model(images=images, texts=texts)      # TODO: Create abstraction that can convert batch keys into model input keys for all models
    output = model(task_key='vqa', **inputs)
    logits = output[1]
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/objectives.py#L317
    loss = loss_criterion(logits, target) * target.shape[1]

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    wandb.log({'vqa': {'loss': loss.item()}})

    return loss.item()