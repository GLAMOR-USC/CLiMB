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

sys.path.insert(0, '.')

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset
from data.visionlanguage_datasets.vqa_dataset import build_vqa_dataloader

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size())#.cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def train_vqa(args, encoder, task_configs, model_config, tokenizer, device):

    vqa_config = task_configs['vqa']
    data_dir = vqa_config['data_dir']
    num_labels = vqa_config['num_labels']
    images_source = vqa_config['images_source']

    mscoco_config = task_configs[images_source]
    images_dataset = MSCOCOImagesDataset(mscoco_config['data_dir'])

    encoder_dim = model_config['encoder_dim']
    visual_mode = model_config['visual_mode']

    # Create model
    classifier_class = model_config['classifier_class']
    model = classifier_class(encoder=encoder, 
                             encoder_dim=encoder_dim, 
                             num_labels=num_labels)
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)

    best_score = 0
    best_model = {
        'epoch': 0,
        'model': copy.deepcopy(model), #model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }

    for epoch in range(num_epochs):
        model.train()
        model.zero_grad()

        # Training loop for epoch
        for step, batch in enumerate(tqdm(vqa_train_dataloader, desc='Training epoch {}'.format(epoch+1))):
            images = batch['images']
            texts = batch['questions']
            target = batch['target_scores']

            output = model(images=images, texts=texts)      # TODO: Create abstraction that can convert batch keys into model input keys for all models
            logits = output[1]
            loss = loss_criterion(logits, target)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            #if (step+1) % 5 == 0:
            #    break

        # Do evaluation after epoch
        eval_score = eval_vqa(args, model, vqa_val_dataloader)
        logger.info("Evaluation after epoch {}: {:.2f}".format(epoch+1, eval_score))
        if eval_score > best_score:
            logger.info("New best evaluation score: {:.2f}".format(eval_score))
            best_score = eval_score
            best_model['epoch'] = epoch
            best_model['model'] = copy.deepcopy(model)

        return best_score, best_model

def eval_vqa(args, model, vqa_val_dataloader):

    model.eval()
    eval_score = 0

    for step, batch in enumerate(tqdm(vqa_val_dataloader, desc='Evaluating on VQA val set')):
        images = batch['images']
        texts = batch['questions']
        target = batch['target_scores']

        output = model(images=images, texts=texts)      # TODO: Create abstraction that can convert batch keys into model input keys for all models
        logits = output[1]

        answer_scores = compute_score_with_logits(logits, target)
        batch_scores = torch.sum(answer_scores, 1)

        eval_score += batch_scores.sum().item()

    eval_score = eval_score/len(vqa_val_dataloader.dataset)*100.0
    return eval_score