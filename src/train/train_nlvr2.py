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

class NLVR2Trainer:

    def __init__(self, args, task_configs, model_config, tokenizer, device):

        self.args = args
        self.tokenizer = tokenizer
        self.device = device

        self.nlvr_config = task_configs['nlvr2']
        self.data_dir = os.path.join(args.mcl_data_dir, self.nlvr_config['data_dir'])

        # Model-specific stuff
        self.visual_mode = model_config['visual_mode']
        self.batch2inputs_converter = model_config['batch2inputs_converter']

        # Create dataloaders for training and validation
        self.nlvr_train_dataloader = build_nlvr2_dataloader(args=args,
                                                    data_dir=self.data_dir,
                                                    split='train',
                                                    visual_mode=self.visual_mode)
    
        self.nlvr_val_dataloader = build_nlvr2_dataloader(args=args,
                                                     data_dir=self.data_dir,
                                                     split='val',
                                                     visual_mode=self.visual_mode)

        # Training hyperparameters
        self.num_epochs = self.nlvr_config['num_epochs']
        self.lr = self.nlvr_config['lr']
        self.adam_epsilon = self.nlvr_config['adam_epsilon']
        self.weight_decay = self.nlvr_config['weight_decay']
        self.loss_criterion = nn.CrossEntropyLoss()
        self.max_steps = len(self.nlvr_train_dataloader) * self.num_epochs
        self.warmup_ratio = 0.1 # TODO remove hard code

    def get_train_dataloader(self):
        return self.nlvr_train_dataloader

    def get_collate_fn(self):
        return self.nlvr_train_dataloader.collate_fn

    def forward_pass(self, model, batch, do_eval=False):

        inputs = self.batch2inputs_converter(batch)
        if do_eval is True:
            with torch.no_grad():
                output = model(task_key='nlvr2', **inputs)
        else:
            output = model(task_key='nlvr2', **inputs)
        return output


    def train_step(self, model, batch, optimizer=None, scheduler=None, ewc=None):

        output = self.forward_pass(model, batch)
        logits = output[1]
        target = batch['labels'].to(self.device)
        loss = self.loss_criterion(logits, target)

        if ewc is not None and ewc.do_ewc() is True:
            ewc_task, ewc_loss = ewc.compute_ewc_loss(model)
            total_loss = loss + ewc_loss
            total_loss.backward()
        else:
            ewc_task = None
            ewc_loss = None
            loss.backward()

        if optimizer is not None:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        return loss, output, ewc_task, ewc_loss

    def create_optimizer(self, model):

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.adam_epsilon, betas=(0.9, 0.98))
        return optimizer

    def train(self, model, replay_memory=None, ewc=None):

        model.to(self.device)
        if self.args.cl_algorithm == 'adapter':
            model.set_active_adapters("nlvr2")
        elif self.args.cl_algorithm == 'experience_replay':
            assert replay_memory is not None
            do_replay = replay_memory.do_replay()
        elif self.args.cl_algorithm == 'ewc':
            assert ewc is not None
            do_ewc = ewc.do_ewc()


        # Create optimizer
        optimizer = self.create_optimizer(model)
        # Create Scheduler
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.max_steps * self.warmup_ratio),
            num_training_steps=self.max_steps,
            lr_end=0,
            power=1,
        )

        best_score = 0
        best_model = {
            'epoch': 0,
            'model': copy.deepcopy(model), #model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }

        model.zero_grad()
        for epoch in range(self.num_epochs):
            # Training loop for epoch

            model.train()
            for step, batch in enumerate(tqdm(self.nlvr_train_dataloader, desc='Training epoch {}'.format(epoch+1))):

                loss, output, ewc_task, ewc_loss = self.train_step(model, batch, optimizer, scheduler, ewc)

                if self.args.cl_algorithm == 'experience_replay' and do_replay is True:
                    if (step + 1) % self.args.replay_frequency == 0:
                        sampled_replay_task = replay_memory.sample_replay_task()
                        replay_loss = replay_memory.run_replay_step(task_key=sampled_replay_task, model=model)

                if (step + 1) % 100 == 0:
                    log_dict = {'nlvr': {'loss': loss.item()}}
                    if ewc is not None and do_ewc is True:
                        log_dict[ewc_task] = {'ewc_loss': ewc_loss.item()}
                    wandb.log(log_dict)

            # Do evaluation after epoch
            eval_score = self.eval(model)
            logger.info("Evaluation after epoch {}: {:.2f}".format(epoch+1, eval_score))
            wandb.log({'nlvr': {'val_score': eval_score}})
            if eval_score > best_score:
                logger.info("New best evaluation score: {:.2f}".format(eval_score))
                best_score = eval_score
                best_model['epoch'] = epoch
                best_model['model'] = copy.deepcopy(model)

        return best_score, best_model

    def eval(self, model):

        model.eval()
        eval_score = 0

        for step, batch in enumerate(tqdm(self.nlvr_val_dataloader, desc='Evaluating on NLVR2 val set')):
            output = self.forward_pass(model, batch, do_eval=True)
            logits = output[1]
            batch_scores = (logits.argmax(-1).cpu() == batch['labels'])
            eval_score += batch_scores.sum().item()

        eval_score = eval_score/len(self.nlvr_val_dataloader.dataset)*100.0

        model.train()
        return eval_score

    def eval_forgetting(self, model, model_path):

        model.to(self.device)
        if self.args.cl_algorithm == 'adapter':
            model.set_active_adapters("nlvr2")

        # Load model with encoder weights from encoder_path, and classifier weights from model_path
        model.load_state_dict(torch.load(model_path))
        logger.info("Loaded model checkpoint from {}".format(model_path))

        return self.eval(model)
