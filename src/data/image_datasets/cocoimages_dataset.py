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
from torch.utils.data import Dataset

from PIL import Image
from utils.image_utils import resize_image

class MSCOCOImagesDataset(Dataset):

    def __init__(self, coco_dir, split, image_size=(384,640)):

        self.images_dir = os.path.join(coco_dir, 'images', '{}2017'.format(split))        # split = {train, val, test}. Using only 2017 image splits.
        self.split = split
        self.image_size = image_size

        image_filenames = os.listdir(self.images_dir)
        self.imageid2filename = {}
        for fn in image_filenames:
            image_id = int(fn.strip('.jpg'))
            self.imageid2filename[image_id] = os.path.join(self.images_dir, fn)
        self.imageids = list(self.imageid2filename.keys())

    def get_image_data(self, image_id, feats_type):

        assert feats_type in ['patch', 'fast-rcnn']
        if feats_type == 'patch':
            return self.get_raw_image_tensor(image_id)
        elif feats_type == 'fast-rcnn':
            raise NotImplementedError("Have not implemented Fast-RCNN feature inputs for MS-COCO images!")

    def get_raw_image_tensor(self, image_id):

        assert image_id in self.imageids
        image_fn = self.imageid2filename[image_id]
        image = Image.open(image_fn)
        image_arr = resize_image(image, self.image_size)
        image_tensor = torch.tensor(image_arr).permute(2, 0, 1).float()
        #if torch.max(image_tensor) == 58.0 or torch.count_nonzero(image_tensor) == 0:
        #    raise Exception("Found an invalid image")
        image.close()
        return image_tensor

if __name__ == '__main__':

    dataset = MSCOCOImagesDataset('/data/datasets/MCL/ms-coco/', 'val')
    imgid = dataset.imageids[0]
    x = dataset.get_image_data(imgid, 'patch')
    print(x.shape)