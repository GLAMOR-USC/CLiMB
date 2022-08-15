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


class iNat2019Dataset(Dataset):

    def __init__(self, data_dir, mode, n_shot=None, subsample_seed=None):
        """
        Initiate the Dataset - loads all the image filenames and the corresponding labels into self.dataset

        data_dir: path containing annotations and images
        mode: either train/val/test
        n_shot: n-shot per class
        subsampled_seed: random seed for low-shot subsampling
        """

        self.data_dir = data_dir
        self.mode = mode
        self.n_shot = n_shot
        self.subsample_seed = subsample_seed
        self.pil_transform = T.Resize(size=384, max_size=640)

        remap_mode = {'train':'train', 'val':'train', 'test':'val'}
        self.annot_file = os.path.join(data_dir, f'{remap_mode[mode]}2019.json')

        self.preprocess()


    def get_train_val_split(self, dataset, val_num_per_class=50):
        """
        Split the validation set from the original training set and do low-shot subsampling

        dataset: the original training set
        val_num_per_class: number of data in the validation set per class
        """

        train_dataset, val_dataset = [], []
        # split each class into train/val; balanced
        for cls_data in dataset:
            # some classes have very few data
            if len(cls_data) <= self.n_shot:
                train_dataset.extend(cls_data)
                continue

            n_train = len(cls_data) - min(int(len(cls_data)*0.1), val_num_per_class)
            # shuffle before train/val split
            random.seed(2022)
            random.shuffle(cls_data)

            train_cls_ds = cls_data[:n_train]
            val_dataset.extend(cls_data[n_train:])

            #subsample n-shot per class in the training set with different seeds
            if self.mode == 'train':
                random.seed(self.subsample_seed)
                random.shuffle(train_cls_ds)
                train_dataset.extend(train_cls_ds[:self.n_shot])

        if self.mode == 'train':
            return train_dataset
        else:
            return val_dataset
            

    def preprocess(self):
        """
        Preprocess train/val/test sets, where we split the val set from the training set
        and use the origianl val set as the test set
        Balance each class in the training set and validation set when doing low-shot sampling
        """

        with open(self.annot_file) as f:
            ann_data = json.load(f)

        all_img_fns = [a['file_name'] for a in ann_data['images']]
        all_labels = [a['category_id'] for a in ann_data['annotations']]

        assert len(all_img_fns) == len(all_labels)
        assert len(set(all_labels)) == 1010

        if self.mode == 'test':
            self.dataset = []
            for label, fn in zip(all_labels, all_img_fns):
                fn = os.path.join(self.data_dir, fn)
                self.dataset.append([fn, label])

        else: # train & val are split from the original training set
            dataset = [[] for _ in range(1010)]
            for label, fn in zip(all_labels, all_img_fns):
                fn = os.path.join(self.data_dir, fn)
                dataset[label].append([fn, label])

            self.dataset = self.get_train_val_split(dataset)

        self.num_images = len(self.dataset)
        logger.info(f'# {self.num_images} images in {self.mode} set')


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


def get_data_loader(args, data_dir, split, n_shot=None, subsampled_seed=None):
    """
    Retrun a torch.utils.data.DataLoader for the dataset

    args: arguments provided by user
    data_dir: path containing annotations and images
    split: either train/val/test split
    n_shot: n-shot per class
    subsampled_seed: random seed for low-shot subsampling
    """

    logger.info(f"Creating iNat2019Dataset {split} dataloader")

    dataset = iNat2019Dataset(data_dir, split, n_shot, subsampled_seed)
    batch_size = args.batch_size if split == 'train' else 128
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers = args.num_workers,
        batch_size = batch_size,
        shuffle = (split=='train'),
        collate_fn = lambda x: batch_collate(x)
        )
    return dataloader
