import os
import sys
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertTokenizer, BertModel
from transformers import ViltProcessor, ViltModel

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class ViltEncoder(nn.Module):

    def __init__(self, processor, vilt, device):
        '''
        Wrapper around Vilt model from huggingface library
        this is the class that gets saved during checkpointing for continual learning
        '''

        super().__init__()
        self.processor = processor
        self.vilt = vilt
        self.device = device

    def process_inputs(self, images, texts):

        encodings = self.processor(images=images, text=texts, padding=True, return_tensors='pt').to(self.device)
        return encodings

    def forward(self, **encodings):

        output = self.vilt(**encodings)
        return output.pooler_output

class ViltForImageTextClassification(nn.Module):

    def __init__(self, encoder, encoder_dim, num_labels):

        '''
        encoder - instance of ViltEncoder class
        encoder_dim - output dimension of vilt encoder
        num_labels - number of labels for classification task
        '''

        super().__init__()

        self.vilt_encoder = encoder
        self.clf_layer = nn.Linear(encoder_dim, num_labels)

    def forward(self, images, texts):

        encodings = self.vilt_encoder.process_inputs(images, texts)
        encoder_output = self.vilt_encoder(**encodings)

        output_logits = self.clf_layer(encoder_output)
        return encoder_output, output_logits

def load_vilt_encoder(pretrained_vilt_name, device):

    logger.info("-"*100)
    logger.info("Loading pretrained ViLT model: {}".format(pretrained_vilt_name))
    vilt_processor = ViltProcessor.from_pretrained(pretrained_vilt_name)
    vilt = ViltModel.from_pretrained(pretrained_vilt_name)
    vilt_encoder = ViltEncoder(vilt_processor, vilt, device)
    logger.info("Successfully loaded pretrained ViLT model")
    return vilt_encoder

def convert_batch_to_model_input_dict(batch):

    return {'images': batch['images'],
            'texts': batch['raw_texts']}