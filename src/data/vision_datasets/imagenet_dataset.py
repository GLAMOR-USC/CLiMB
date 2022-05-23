import sys
import pdb
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

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)


class ImageNetDataset(Dataset):

    def __init__(self, image_dir, mode, n_shot=None, subsample_seed=None):
        self.image_dir = image_dir
        self.mode = mode
        self.n_shot = n_shot
        self.subsample_seed = subsample_seed
        self.pil_transform = T.Resize(size=384, max_size=640)

        self.preprocess()


    def get_train_val_split(self, dataset, split_ratio=0.9):
        train_dataset, val_dataset = [], []
        # split each class into train/val; balanced
        for cls_data in dataset:
            n_train = int(len(cls_data)*split_ratio)
            # shuffle before train/val split
            random.seed(2022)
            random.shuffle(cls_data)

            train_cls_ds = cls_data[:n_train]
            val_dataset.extend(cls_data[n_train:])

            if self.mode != 'train': continue
            if self.n_shot is None:
                train_dataset.extend(train_cls_ds)
            else: #subsample the training set with different seeds
                random.seed(self.subsample_seed)
                random.shuffle(train_cls_ds)
                train_dataset.extend(train_cls_ds[:self.n_shot])

        if self.mode == 'train':
            return train_dataset
        else:
            return val_dataset
            

    def preprocess(self):
        all_classes = os.listdir(self.image_dir)

        dataset = [[] for _ in range(len(all_classes))]
        n_imgs = 0
        for label, dir_name in enumerate(all_classes):
            filenames = glob.glob(os.path.join(self.image_dir, dir_name, '*.JPEG'))
            for fn in filenames:
                dataset[label].append([fn, label])
                n_imgs += 1

        self.dataset = self.get_train_val_split(dataset)
        self.num_images = len(self.dataset)
        print(f'# {self.num_images} images in {self.mode} set')


    def __getitem__(self, index):
        filename, label = self.dataset[index]
        image = Image.open(filename)
        image = image.convert('RGB')
        if min(list(image.size)) > 384:
            image = self.pil_transform(image)

        return image, label

    def __len__(self):
        return self.num_images


def batch_collate(batch):
    pil_objs, labels = zip(*batch)
    raw_texts = ['This is an image.' for _ in range(len(labels))]
    return {'raw_texts': raw_texts, 
            'images': pil_objs, 
            'labels': torch.LongTensor(labels)}    


def get_data_loader(args, img_dir, split, n_shot=None, subsampled_seed=None):
    logger.info(f"Creating ImageNet {split} dataloader")

    dataset = ImageNetDataset(img_dir, split, n_shot, subsampled_seed)
    batch_size = args.batch_size if split == 'train' else 128
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers = args.num_workers,
        batch_size = batch_size,
        shuffle = (split=='train'),
        collate_fn = lambda x: batch_collate(x)
        )
    return dataloader


'''
if __name__ == '__main__':

    img_dir = '/data/datasets/MCL/ILSVRC2012/train_256'

    class Args:
        def __init__(self):
            self.batch_size = 32
            self.num_workers = 0

    args = Args()
    dataloader = get_data_loader(args, img_dir, 'train', n_shot=64, subsampled_seed=20)
    for batch in tqdm(dataloader):
        print(batch['raw_texts'])
        print(batch['images'])
        print(batch['labels'])
'''
