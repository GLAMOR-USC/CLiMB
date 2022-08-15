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

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset

from PIL import Image
from utils.image_utils import resize_image

class Flickr30KImagesDataset(Dataset):

    def __init__(self, flickr_dir: str, visual_input_type: str, image_size=(384,640)):

        '''
        Initializes a Flickr30KImagesDataset instance that handles image-side processing for SNLI-VE and other tasks that use Flickr images
        coco_dir: directory that contains Flickr30K data (images within 'flickr30k_images' folder)
        visual_input_type: format of visual input to model
        image_size: tuple indicating size of image input to model
        '''

        self.images_dir = os.path.join(flickr_dir, 'flickr30k_images')          # Images across all 2017 splits stored in same directory
        self.image_size = image_size
        self.visual_input_type = visual_input_type
        assert visual_input_type in ['pil-image', 'raw', 'fast-rcnn']

        image_filenames = os.listdir(self.images_dir)
        self.imageid2filename = {}
        for fn in image_filenames:
            image_id = int(fn.strip('.jpg'))
            self.imageid2filename[image_id] = os.path.join(self.images_dir, fn)
        self.imageids = list(self.imageid2filename.keys())

        self.raw_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),                                 # [0, 1]
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1]
        ])

        self.pil_transform = T.Resize(image_size)

    def get_image_data(self, image_id: str):

        '''
        Returns image data according to required visual_input_type. Output format varies by visual_input_type
        '''

        if self.visual_input_type == 'pil-image':
            return self.get_pil_image(image_id)
        if self.visual_input_type == 'raw':
            return self.get_raw_image_tensor(image_id)
        elif self.visual_input_type == 'fast-rcnn':
            raise NotImplementedError("Have not implemented Fast-RCNN feature inputs for Flickr30K images!")

    def get_pil_image(self, image_id: str) -> Image:
        '''
        Loads image corresponding to image_id, re-sizes and returns PIL.Image object
        '''

        assert image_id in self.imageid2filename.keys()
        image_fn = self.imageid2filename[image_id]
        image = Image.open(image_fn)
        image = image.convert('RGB')
        if min(list(image.size)) > 384:
            image = self.pil_transform(image)
        return image

    def get_raw_image_tensor(self, image_id: str) -> torch.Tensor:
        '''
        Loads image corresponding to image_id, re-sizes, and returns tensor of size (3, W, H)
        '''

        assert image_id in self.imageid2filename.keys()
        image_fn = self.imageid2filename[image_id]
        image = Image.open(image_fn)
        image = image.convert('RGB')

        image_tensor = self.raw_transform(image)

        image.close()
        return image_tensor         # (B, 3, W, H)

if __name__ == '__main__':

    dataset = Flickr30KImagesDataset('/data/datasets/MCL/flickr30k/', 'raw')
    imgid = dataset.imageids[0]
    x = dataset.get_image_data(imgid)
    print(x.shape)
