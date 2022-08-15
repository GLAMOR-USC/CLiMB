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
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)



class CocoClsDataset(Dataset):

    def __init__(self, data_dir, mode, n_shot=None, subsample_seed=None):
        """
        Initiate the Dataset - loads all the image filenames and the corresponding labels into self.dataset

        data_dir: path containing annotations and images
        mode: either train/val/test
        n_shot (float): ratio for low-shot subsampling
        subsampled_seed: random seed for low-shot subsampling
        """

        self.data_dir = data_dir
        self.mode = mode
        self.n_shot = n_shot
        self.subsample_seed = subsample_seed
        self.pil_transform = T.Resize(size=384, max_size=640)

        self.images_dir = os.path.join(data_dir, 'images')
        remap_mode = {'train':'train', 'val':'train', 'test':'val'}
        self.fn_mode = remap_mode[mode] 
        self.annot_file = os.path.join(data_dir, 'detections', 'annotations', f'instances_{self.fn_mode}2017.json')

        self.preprocess()


    def get_train_val_split(self, dataset, val_ratio=0.1):
        """
        Split the validation set from the original training set and do low-shot subsampling

        dataset: the original training set
        val_ratio: the ratio to split the validation set
        """

        train_dataset, val_dataset = [], []
        # shuffle before train/val split
        random.seed(2022)
        random.shuffle(dataset)
        n_val = int(len(dataset)*val_ratio)

        if self.mode == 'val': 
            val_dataset = dataset[: n_val]
            return val_dataset

        train_dataset = dataset[n_val: ]
        #subsample the training set with different seeds
        random.seed(self.subsample_seed)
        random.shuffle(train_dataset)
        n_train = int(self.n_shot * len(dataset))
        assert n_train < len(train_dataset)
        train_dataset = train_dataset[: n_train]

        return train_dataset


    def preprocess(self):
        """
        Preprocess train/val/test sets, where we split the val set from the training set
        and use the origianl val set as the test set
        We formulate the task as a multi-label object classification task
        """

        cached_fn = os.path.join(self.data_dir, f'cached_{self.fn_mode}.pkl')
        try:
            with open(cached_fn, 'rb') as f:
                dataset = pickle.load(f)
            logger.info(f"Loaded cached file from {cached_fn}!")
        except FileNotFoundError:
            coco = COCO(self.annot_file)
            cat_ids = sorted(list(coco.catToImgs.keys()))
            cat2cls = {cat_i: i for i, cat_i in enumerate(cat_ids)}
            img2classes = defaultdict(list)

            for cat_i, img_ids in coco.catToImgs.items():
                cls_i = cat2cls[cat_i]
                img_ids = set(img_ids)
                for img_i in img_ids:
                    img2classes[img_i].append(cls_i)

            sorted_img_ids = sorted(img2classes.keys())
            dataset = []
            for img_i in sorted_img_ids:
                img_fn = os.path.join(self.images_dir, "{:012d}.jpg".format(img_i))
                class_ids = img2classes[img_i] # list
                dataset.append([img_fn, class_ids])

            with open(cached_fn, 'wb') as f:
                pickle.dump(dataset, f)
            logger.info(f"Dumped cached file to {cached_fn}!")

        if self.mode == 'test':
            self.dataset = dataset
        else:
            self.dataset = self.get_train_val_split(dataset)

        self.num_images = len(self.dataset)
        logger.info(f'# {self.num_images} images in {self.mode} set')


    def __getitem__(self, i):
        labels = torch.zeros(80, dtype=torch.float)
        filename, class_ids = self.dataset[i]
        labels[class_ids] = 1
        image = Image.open(filename)
        image = image.convert('RGB')
        if min(list(image.size)) > 384:
            image = self.pil_transform(image)

        return image, labels

    def __len__(self):
        return self.num_images


def batch_collate(batch):
    pil_objs, labels = zip(*batch)
    raw_texts = ['This is an image.' for _ in range(len(labels))]
    return {'raw_texts': raw_texts, 
            'images': pil_objs, 
            'labels': torch.stack(labels)}


def get_data_loader(args, data_dir, split, n_shot=None, subsampled_seed=None):
    """
    Retrun a torch.utils.data.DataLoader for the dataset

    args: arguments provided by user
    data_dir: path containing annotations and images
    split: either train/val/test split
    n_shot (float): ratio for low-shot subsampling
    subsampled_seed: random seed for low-shot subsampling
    """

    logger.info(f"Creating COCO-classification {split} dataloader")

    dataset = CocoClsDataset(data_dir, split, n_shot, subsampled_seed)
    batch_size = args.batch_size if split == 'train' else 128
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers = args.num_workers,
        batch_size = batch_size,
        shuffle = (split=='train'),
        collate_fn = lambda x: batch_collate(x)
        )
    return dataloader

if __name__ == '__main__':
    pass

