import sys
import pdb
import os
import time
import json
import csv
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

    def __init__(self, data_dir, mode, n_shot=None, subsample_seed=None):
        self.data_dir = data_dir
        self.mode = mode
        self.n_shot = n_shot
        self.subsample_seed = subsample_seed
        self.pil_transform = T.Resize(size=384, max_size=640)

        if self.mode == 'test':
            self.preprocess_test()
        else:
            self.preprocess_train_val()

        self.num_images = len(self.dataset)
        logger.info(f'# {self.num_images} images in {self.mode} set')


    def get_train_val_split(self, dataset, val_num_per_class=50):
        train_dataset, val_dataset = [], []
        # split each class into train/val; balanced
        for cls_data in dataset:
            n_train = len(cls_data) - val_num_per_class
            # shuffle before train/val split
            random.seed(2022)
            random.shuffle(cls_data)

            train_cls_ds = cls_data[:n_train]
            val_dataset.extend(cls_data[n_train:])

            # subsample n-shot per class in the training set with different seeds
            if self.mode == 'train':
                random.seed(self.subsample_seed)
                random.shuffle(train_cls_ds)
                train_dataset.extend(train_cls_ds[:self.n_shot])

        if self.mode == 'train':
            return train_dataset
        else:
            return val_dataset


    def preprocess_train_val(self):
        self.image_dir = os.path.join(self.data_dir, 'train')
        all_classes = sorted(os.listdir(self.image_dir))
        assert len(all_classes) == 1000

        dataset = [[] for _ in range(len(all_classes))]
        n_imgs = 0
        for label, dir_name in enumerate(all_classes):
            filenames = glob.glob(os.path.join(self.image_dir, dir_name, '*.JPEG'))
            for fn in filenames:
                dataset[label].append([fn, label])
                n_imgs += 1

        self.dataset = self.get_train_val_split(dataset)
            

    def preprocess_test(self):
        self.image_dir = os.path.join(self.data_dir, 'val')
        annot_file = os.path.join(self.data_dir, "LOC_val_solution.csv")

        all_classes = sorted(os.listdir(os.path.join(self.data_dir, 'train')))
        assert len(all_classes) == 1000
        dir2lb = {name:i for i, name in enumerate(all_classes)}

        self.dataset = []
        lines = csv.DictReader(open(annot_file))
        for line in lines:
            fn = os.path.join(self.data_dir, 'val', line['ImageId']+".JPEG")
            cls_dir = line['PredictionString'].split()[0]
            label = dir2lb[cls_dir]
            self.dataset.append([fn, label])


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
    logger.info(f"Creating ImageNet {split} dataloader")

    dataset = ImageNetDataset(data_dir, split, n_shot, subsampled_seed)
    batch_size = args.batch_size if split == 'train' else 128
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers = args.num_workers,
        batch_size = batch_size,
        shuffle = (split=='train'),
        collate_fn = lambda x: batch_collate(x)
        )
    return dataloader
