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
from tqdm import tqdm

import transformers
from transformers.adapters import AdapterConfig
from transformers import PfeifferConfig, HoulsbyConfig, ParallelConfig, CompacterConfig


from modeling.continual_learner import EncoderWrapper, ContinualLearner

logger = logging.getLogger(__name__)


ADAPTER_MAP = {
    'pfeiffer': PfeifferConfig,
    'houlsby': HoulsbyConfig,
    'parallel': ParallelConfig,
    'compacter': CompacterConfig,
}

SUPPORTED_ADAPTER_METHODS = ['vanilla']

class AdapterHandler:

    def __init__(self, adapter_method, args):

        self.args = args
        self.adapter_method = adapter_method
        
        adapter_config = AdapterConfig.load(args.adapter_config)
        config_dict = adapter_config.to_dict()
        if args.adapter_reduction_factor > 0:
            config_dict['reduction_factor'] = args.adapter_reduction_factor
        self.adapter_config = AdapterConfig.from_dict(config_dict)
        
        logger.info("Adding Adapter layers with configuration:")
        logger.info(str(adapter_config))

    def add_adapters_to_model(self, model: ContinualLearner):

        for task_key in self.args.ordered_cl_tasks:
            model.add_adapter(task_key, config=self.adapter_config)
        logger.info("Added Adapters for tasks: {}".format(','.format(self.args.ordered_cl_tasks)))

    def activate_adapter_for_training(self, task_key: str, model: ContinualLearner):

        model.train_adapter(task_key)
        model.set_active_adapters(task_key)

    def activate_adapter_for_eval(self, task_key: str, model: ContinualLearner):

        model.set_active_adapters(task_key)