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

class MSCOCOImagesDataset(Dataset):

    def __init__(self, coco_dir, image_size=(224,224)):

        self.images_dir = os.path.join(coco_dir, 'images')          # Images across all 2017 splits stored in same directory
        self.image_size = image_size

        image_filenames = os.listdir(self.images_dir)
        self.imageid2filename = {}
        for fn in image_filenames:
            image_id = int(fn.strip('.jpg'))
            self.imageid2filename[image_id] = os.path.join(self.images_dir, fn)
        self.imageids = list(self.imageid2filename.keys())

        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),                                 # [0, 1]
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1]
        ])

    def get_image_data(self, image_id, feats_type):

        assert feats_type in ['raw', 'fast-rcnn']
        if feats_type == 'raw':
            return self.get_raw_image_tensor(image_id)
        elif feats_type == 'fast-rcnn':
            raise NotImplementedError("Have not implemented Fast-RCNN feature inputs for MS-COCO images!")

    def get_raw_image_tensor(self, image_id):

        assert image_id in self.imageids
        image_fn = self.imageid2filename[image_id]
        image = Image.open(image_fn)

        #image_arr = resize_image(image, self.image_size)
        #image_tensor = torch.tensor(image_arr).permute(2, 0, 1).float()
        image_tensor = self.transform(image)

        #if torch.max(image_tensor) == 58.0 or torch.count_nonzero(image_tensor) == 0:
        #    raise Exception("Found an invalid image")
        image.close()
        return image_tensor         # (B, 3, W, H)

if __name__ == '__main__':

    dataset = MSCOCOImagesDataset('/data/datasets/MCL/ms-coco/')
    imgid = dataset.imageids[0]
    x = dataset.get_image_data(imgid, 'raw')
    print(x.shape)