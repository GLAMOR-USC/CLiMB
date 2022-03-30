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
from PIL import Image
import copy
import pdb
import wandb
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup

from data.language_datasets.text_dataset import get_data_loader

sys.path.insert(0, '.')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["WANDB_START_METHOD"] = "thread"
#wandb.init(project='language')

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

def train_language(args, encoder, task_config, model_config, tokenizer, device):
    task_name = task_config['task_name']
    data_dir = task_config['data_dir']
    num_labels = task_config['num_labels']
    max_len = task_config['max_len']
    cache_dir = task_config['cache_dir']

    # Create model
    from modeling.vilt_modeling import convert_language_batch_to_model_input_dict #TODO
    batch2inputs_converter = convert_language_batch_to_model_input_dict
    encoder_dim = model_config['encoder_dim']
    visual_mode = model_config['visual_mode']
    classifier_class = model_config['classifier_class']
    model = classifier_class(encoder=encoder, 
                             encoder_dim=encoder_dim, 
                             num_labels=num_labels,
                             num_images=1)
    model.to(device)

    # Create dataloaders for training and validation
    train_dataloader = get_data_loader(tokenizer, 
        task_name=task_name, 
        split='train', 
        max_len=max_len, 
        batch_size=args.batch_size, 
        cache_dir=cache_dir)

    val_dataloader = get_data_loader(tokenizer, 
        task_name=task_name, 
        split='val', 
        max_len=max_len, 
        batch_size=args.batch_size*2, 
        cache_dir=cache_dir)

    # Training hyperparameters
    num_epochs = task_config['num_epochs']
    lr = task_config['lr']
    adam_epsilon = task_config['adam_epsilon']
    weight_decay = task_config['weight_decay']
    warmup_ratio = task_config['warmup_ratio']

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

    # load the pre-computed mean image
    image_fn = "coco_mean_image.png"
    mean_image = Image.open(image_fn)

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
            target = batch[1].to(device)
            inputs = batch2inputs_converter(batch[0], mean_image)

            output = model(**inputs)
            logits = output[1]
            loss = loss_criterion(logits, target)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print('loss:', loss.item())

        # Do evaluation after epoch
        eval_score = eval(args, model, mean_image, val_dataloader, device, batch2inputs_converter)
        logger.info("Evaluation after epoch {}: {:.2f}".format(epoch+1, eval_score))

        if eval_score > best_score:
            logger.info("New best evaluation score: {:.2f}".format(eval_score))
            best_score = eval_score
            best_model['epoch'] = epoch
            best_model['model'] = copy.deepcopy(model)

    return best_score, best_model


def eval(args, model, mean_image, val_dataloader, device, batch2inputs_converter):

    model.eval()
    eval_score = 0
    for step, batch in enumerate(tqdm(val_dataloader, desc='Evaluating on val set')):
        labels = batch[1]
        inputs = batch2inputs_converter(batch[0], mean_image)
        with torch.no_grad():
            output = model(**inputs)
            logits = output[1]

        batch_scores = (logits.argmax(-1).cpu() == labels)
        eval_score += batch_scores.sum().item()

    eval_score = eval_score/len(val_dataloader.dataset)*100.0
    logger.info(f'Eval_acc: {eval_score:.3f}')

    model.train()
    return eval_score
