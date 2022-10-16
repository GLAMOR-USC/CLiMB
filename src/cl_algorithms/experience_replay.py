import argparse
import random
import logging
from typing import List, Dict

import torch
from torch import nn
from torch.optim import AdamW

from modeling.continual_learner import ContinualLearner
from train.visionlanguage_tasks.task_trainer import TaskTrainer
from utils.wandb import wandb_logger

logger = logging.getLogger(__name__)


class ExperienceReplayMemory:

    def __init__(self):
        '''
        Initializes ER memory with empty memory buffer dict
        '''
        self.memory_buffers = {}

    def add_task_memory_buffer(self, 
                               args: argparse.Namespace, 
                               task_key: str, 
                               task_config: Dict, 
                               task_trainer: TaskTrainer, 
                               memory_percentage: float, 
                               sampling_strategy: str):
        '''
        Creates a memory buffer for new task
        '''

        task_buffer = TaskMemoryBuffer(args, task_key, task_config, task_trainer, memory_percentage, sampling_strategy)
        self.memory_buffers[task_key] = task_buffer

    def do_replay(self) -> bool:
        '''
        Return true if there are any tasks in the memory to do replay on, else False
        '''
        return True if len(self.memory_buffers) > 0 else False

    def sample_replay_task(self) -> str:
        '''
        Samples a previous task at random
        '''
        previous_tasks = list(self.memory_buffers.keys())
        sampled_previous_task = random.choice(previous_tasks)
        return sampled_previous_task

    def run_replay_step(self, task_key: str, model: ContinualLearner) -> torch.Tensor:
        '''
        Performs a single training step on previous task, by sampling a batch from task bugger
        '''
        task_buffer = self.memory_buffers[task_key]
        task_config = task_buffer.task_config
        task_trainer = task_buffer.task_trainer

        optimizer = model.create_optimizer(task_trainer.hparams)
        replay_batch = task_buffer.sample_replay_batch()
        replay_loss, output, _, _ = task_trainer.train_step(model, replay_batch, optimizer)

        logger.info("{} replay step: loss = {:.5f}".format(task_config['task_name'], replay_loss.item()))
        wandb_logger.log({task_key: {'loss': replay_loss.item()}})
        return replay_loss

class TaskMemoryBuffer:

    '''
    Buffer of training examples that can be used for replay steps
    '''
    def __init__(self, 
                 args: argparse.Namespace, 
                 task_key: str, 
                 task_config: Dict, 
                 task_trainer: TaskTrainer, 
                 memory_percentage: float, 
                 sampling_strategy: str):

        '''
        Creates a memory buffer for new task, which samples a small percentage of training data for experience replay
        '''

        self.task_key = task_key
        self.task_name = task_config['task_name']
        self.task_config = task_config

        self.task_trainer = task_trainer
        self.dataset = task_trainer.get_train_dataloader().dataset
        self.batch_collate_fn = task_trainer.get_collate_fn()
        if task_key == 'nlvr2':
            self.batch_size = int(args.batch_size/2)
        elif task_key == 'vcr':
            self.batch_size = int(args.batch_size/4)
        else:
            self.batch_size = args.batch_size

        self.memory_percentage = memory_percentage                      # Percent of training samples to store in memory
        assert self.memory_percentage < 1.0
        self.memory_size = int(memory_percentage*len(self.dataset))     # Number of training samples that are stored in memory
        self.sampling_strategy = sampling_strategy
        assert sampling_strategy in ['random']                      # Only random sampling for memory buffer implemented so far

        if self.sampling_strategy == 'random':
            train_idxs = list(range(len(self.dataset)))
            self.memory_idxs = random.sample(train_idxs, self.memory_size)

        elif self.sampling_strategy == 'random-balanced':
            raise NotImplementedError("Label-balanced sampling of replay memory is not yet implemented!")

        logger.info("Created {} replay memory buffer, with {} samples in the memory".format(self.task_name, len(self.memory_idxs)))

    def __len__(self):
        return len(self.memory_idxs)

    def sample_replay_batch(self) -> Dict:

        sampled_instances = random.sample(self.memory_idxs, self.batch_size)
        batch = self.batch_collate_fn([self.dataset[i] for i in sampled_instances])
        return batch
