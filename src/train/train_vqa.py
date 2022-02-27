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

sys.path.insert(0, '.')

import numpy as np
import torch
from tqdm import tqdm

from data.images_dataset.cocoimages_dataset import MSCOCOImagesDataset
from data.visionlanguage_datasets.vqa_dataset import build_vqa_dataloader


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
    vqa_train_dataloader = build_vqa_dataloader(data_dir=data_dir,
                                                images_dataset=images_dataset,
                                                split='train',
                                                tokenizer=tokenizer,
                                                visual_mode=visual_mode)

    vqa_val_dataloader = build_vqa_dataloader(data_dir=data_dir,
                                                images_dataset=images_dataset,
                                                split='val',
                                                tokenizer=tokenizer,
                                                visual_mode=visual_mode)

    # Training hyperparameters
    num_epochs = vqa_config['num_epochs']
    lr = vqa_config['lr']
    adam_epsilon = vqa_config['adam_epsilon']
    weight_decay = vqa_config['weight_decay']

    loss_criterion = nn.BCEWithLogitsLoss(reduce='mean')
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)

    for epoch in range(num_epochs):
        model.train()

        for step, batch in enumerate(vqa_train_dataloader):
            images = batch['images']
            texts = batch['questions']
            target = batch['target_scores']

            output = model(images=images, texts=texts)
            logits = output[1]
            loss = loss_criterion(logits, target)

            loss.backward()