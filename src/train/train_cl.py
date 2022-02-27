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

sys.path.insert(0, '.')

import numpy as np
import torch
from tqdm import tqdm
import wandb

from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import ViltProcessor, ViltModel

from modeling.vilt_modeling import ViltEncoder, ViltForSequenceClassification
from train.train_vqa import train_vqa
from configs.model_configs import model_configs

logger = logging.getLogger(__name__)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

config_file = '../configs/task_configs.yaml'
task_configs = yaml.load(open(config_file, 'r'))

task_name = 'vqa'
encoder_name = 'vilt'
pretrained_vilt_name = 'dandelin/vilt-b32-mlm'

model_config = model_configs[encoder_name]
# Need to write a method in modeling/ that loads the pretrained model depending on the model type (encoder_name)
vilt_processor = ViltProcessor.from_pretrained(pretrained_vilt_name)
vilt = ViltModel.from_pretrained(pretrained_vilt_name)

vilt_encoder = ViltEncoder(vilt_processor, vilt, device)


task_config = task_configs['vqa']


train_vqa(args, vilt_encoder, task_config, model_config, tokenizer, device)