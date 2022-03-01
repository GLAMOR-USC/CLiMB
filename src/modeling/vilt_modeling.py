import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertTokenizer, BertModel
from transformers import ViltProcessor, ViltModel

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

    vilt_processor = ViltProcessor.from_pretrained(pretrained_vilt_name)
    vilt = ViltModel.from_pretrained(pretrained_vilt_name)
    vilt_encoder = ViltEncoder(vilt_processor, vilt, device)
    return vilt_encoder
