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
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
from utils.image_utils import resize_image
from torchvision.utils import save_image
from torch.utils import data

class MSCOCOImagesDataset(Dataset):

    def __init__(self, coco_dir, image_size=(384, 384)):

        self.images_dir = os.path.join(coco_dir, 'images')          # Images across all 2017 splits stored in same directory
        self.image_size = image_size

        image_filenames = os.listdir(self.images_dir)
        self.imageid2filename = {}
        for fn in image_filenames:
            image_id = int(fn.strip('.jpg'))
            self.imageid2filename[image_id] = os.path.join(self.images_dir, fn)
        self.imageids = list(self.imageid2filename.keys())
        self.num_images = len(self.imageids)

        self.raw_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),                                 # [0, 1]
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1]
        ])


    def get_raw_image_tensor(self, image_fn):
        image = Image.open(image_fn)
        image = image.convert('RGB')
        image_tensor = self.raw_transform(image)
        image.close()
        return image_tensor         # (B, 3, W, H)

    def __getitem__(self, i):
        image_id = self.imageids[i]
        image_fn = self.imageid2filename[image_id]
        x = self.get_raw_image_tensor(image_fn)
        return x

    def __len__(self):
        return self.num_images


def save_imgs(image, fn, n=4):
    def denorm(x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    imgs = denorm(image[:n].cpu())
    save_image(imgs, f'{fn}.png', nrow=n, padding=0)
    print(f'Save {fn}.png!', imgs.shape)


if __name__ == '__main__':

    dataset = MSCOCOImagesDataset('/data/datasets/MCL/ms-coco/')
    data_loader = data.DataLoader(dataset=dataset,
                              batch_size=2048,
                              shuffle=False,
                              drop_last=False,
                              num_workers=8,
                              )
    
    print('# images:', dataset.num_images)
    sum_image = None
    for batch in tqdm(data_loader):
        batch = batch.cuda()
        if sum_image is not None:
            sum_image += batch.sum(0)
        else:
            sum_image = batch.sum(0)
            tmp_mean_img = batch[:4].sum(0) / 4
            tmp_cat = torch.cat((batch[:4], tmp_mean_img.unsqueeze(0)), 0)
            save_imgs(tmp_cat, 0, 5)
    pdb.set_trace()
    mean_image = sum_image / dataset.num_images
    save_imgs(mean_image.unsqueeze(0), 'coco_mean_image', 1)
