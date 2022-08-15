
import sys
import os
import time
import json
import logging
import random
import glob
import base64
from tqdm import tqdm
from collections import defaultdict
import pickle as pkl
import pdb
from typing import List, Dict

from transformers import BertTokenizer

from PIL import Image
from utils.image_utils import resize_image
from utils.vqa_utils import get_score, target_tensor

from data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset

ALLOWED_VISUAL_INPUT_TYPES = ['raw',                    # A raw tensor of size (3, W, H)
                              'pil-image',              # A PIL.Image instance
                              'fast-rcnn'               # A set of R features, each of dim H
                              ]

def image_collate(images: List, 
                  visual_input_type: str):

    """
    Converts list of B images into a batched image input

    Args:
    images: list of B images - type(image) can vary according to visual_input_type (see ALLOWED_VISUAL_INPUT_TYPES above)
    visual_input_type: one element from ALLOWED_VISUAL_INPUT_TYPES

    Returns:
    collated_images: list/Tensor that contains all the images collated together into a batch input
    """

    if visual_input_type == 'pil-image':
        # returns list of PIL.Image objects
        collated_images = images

    if visual_input_type == 'raw':
        # Stacks individual raw image tensors to give (B, 3, W, H) tensor
        collated_images = torch.stack(images, dim=0)

    elif visual_input_type == 'fast-rcnn':
        # Stack the image tensors, doing padding if necessary for the sequence of region features
        # Each element is a tensor of shape [R_i, H], returns tensor of shape [B, max(R_i), H]
        max_len = max([t.shape[0] for t in images])
        image_tensors_padded = []
        for i in range(len(images)):
            padding_tensor = torch.zeros(max_len-images[i].shape[0], images[i].shape[1])
            padded_tensor = torch.cat((images[i], padding_tensor), dim=0)
            assert padded_tensor.shape[0] == max_len
            image_tensors_padded.append(padded_tensor)
        collated_images = torch.stack(image_tensors_padded, dim=0)        # Pads region features with 0 vectors to give (B, R, hv) tensor

    return collated_images

