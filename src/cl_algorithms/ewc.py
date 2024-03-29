import argparse
import random
from collections import defaultdict
from tqdm import tqdm
import logging
from typing import List, Dict

import torch

from configs.task_configs import task_configs
from modeling.continual_learner import ContinualLearner
from train.visionlanguage_tasks.task_trainer import TaskTrainer

logger = logging.getLogger(__name__)

class EWC:

    def __init__(self, args: argparse.Namespace):
        '''
        Initializes an EWC object with EWC parameters and empty dictionaries that will be used for storing model parameters
        '''
        self.fisher_sample_percentage = args.ewc_fisher_sample_percentage
        self.ewc_loss_weight = args.ewc_loss_weight
        self.fisher_dict = {}
        self.param_dict = {}
        self.task_keys = []

    def save_task_parameters(self, 
                             task_key: str, 
                             model: ContinualLearner, 
                             task_trainer: TaskTrainer, 
                             device: torch.device):
        '''
        Saves model parameters after training on a task, and computes Fisher information matrix
        '''

        task_config = task_configs[task_key]
        self.fisher_dict[task_key] = defaultdict(float)

        # Save model params
        self.param_dict[task_key] = {}
        for name, param in model.get_encoder().named_parameters():
            self.param_dict[task_key][name] = param.data.cpu().clone()

        assert task_key not in self.task_keys
        self.task_keys.append(task_key)

        optimizer = model.create_optimizer(task_trainer.hparams)
        batch2inputs_converter = task_trainer.batch2inputs_converter
        dataloader = task_trainer.get_train_dataloader()
        loss_criterion = task_trainer.loss_criterion
        fisher_sample_size = int(self.fisher_sample_percentage*len(dataloader.dataset))
        self.device = task_trainer.device

        model.to(device)
        optimizer.zero_grad()
        num_samples_completed = 0
        # Create fisher matrix
        for step, batch in enumerate(tqdm(dataloader, desc='Computing Fisher information matrix for {} checkpoint'.format(task_config['task_name']))):
            loss, output, _, _ = task_trainer.train_step(model, batch)

            for name, param in model.get_encoder().named_parameters():
                if param.grad is not None:
                    self.fisher_dict[task_key][name] += param.grad.data.pow(2).cpu().clone()

            num_samples_completed += len(batch['raw_texts'])
            if num_samples_completed >= fisher_sample_size:
                break

        for name in self.fisher_dict[task_key].keys():
            self.fisher_dict[task_key][name] /=  num_samples_completed

        logger.info("Saved encoder parameters for {} task!".format(task_config['task_name']))

    def compute_ewc_loss(self, model: ContinualLearner) -> (str, torch.Tensor):
        '''
        Randomly samples previous task, and computes EWC loss by comparing model parameters with previous parameters
        '''

        ewc_task_key = random.choice(self.task_keys)
        ewc_loss = 0
        for name, param in model.get_encoder().named_parameters():
            if name in self.fisher_dict[ewc_task_key].keys():
                ewc_param = self.param_dict[ewc_task_key][name].to(self.device)
                fisher_info = self.fisher_dict[ewc_task_key][name].to(self.device)
                ewc_loss += (fisher_info*((param - ewc_param).pow(2))).sum()
        return ewc_task_key, self.ewc_loss_weight*ewc_loss

    def do_ewc(self):
        return True if len(self.task_keys) > 0 else False