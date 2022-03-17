import argparse
import time
import numpy as np
import logging
import pdb
from tqdm import tqdm
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import os
from PIL import Image

## image_dataset -- tejas
from data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class MSCOCODetection(datasets.coco.CocoDetection):
    def __init__(self, annotation_dir, split, images_dataset, visual_mode):
        from pycocotools.coco import COCO
        self.images_dataset = images_dataset
        self.transforms = self.images_dataset.raw_transform
        self.visual_mode = visual_mode

        coco_cached_file = os.path.join(annotation_dir, 'cached_data', 'coco_detection_{}.pkl'.format(split))
        if os.path.isfile(coco_cached_file):
            # Load image IDs and annotations from cached file
            data = pkl.load(open(coco_cached_file, 'rb'))
            self.image_ids = data['image_ids']
            self.annotations = data['annotations']
            logger.info("Finished loading all the COCO annotations into memory")
        else:
            annFile = os.path.join(annotation_dir, 'instances_{}2017.json'.format(split))
            self.coco = COCO(annFile)
            self.ids = list(self.coco.imgs.keys())
            
            self.cat2cat = dict()
            for cat in self.coco.cats.keys():
                self.cat2cat[cat] = len(self.cat2cat)
            # print(self.cat2cat)
            
            coco = self.coco
            image_ids = []
            annotations = []
            logger.info("Loading COCO information into memory...")

            for index in self.ids:
                # Own coco file
                coco = self.coco
                # Image ID
                img_id = index #self.ids[index]
                ann_ids = coco.getAnnIds(imgIds=img_id)
                target = coco.loadAnns(ann_ids)
                output = torch.zeros((80), dtype=torch.float)
                for obj in target:
                    output[self.cat2cat[obj['category_id']]] = 1
                target = output
                '''
                output = torch.zeros((3, 80), dtype=torch.long)
                for obj in target:
                    if obj['area'] < 32 * 32:
                        output[0][self.cat2cat[obj['category_id']]] = 1
                    elif obj['area'] < 96 * 96:
                        output[1][self.cat2cat[obj['category_id']]] = 1
                    else:
                        output[2][self.cat2cat[obj['category_id']]] = 1
                target = output
                '''
                image_ids.append(img_id)
                annotations.append(target)
            self.image_ids   = image_ids
            self.annotations = annotations
            logger.info("Finished loading all the COCO annotations into memory")

            # Cache the annotations
            cache_data = {
                        'image_ids': self.image_ids,
                        'annotations': self.annotations
                        }
            pkl.dump(cache_data, open(coco_cached_file, 'wb'))

        logger.info("Loaded COCO detection {} set with {} examples".format(split, len(self.image_ids)))

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = self.images_dataset.get_image_data(image_id, self.visual_mode)
        return image, self.annotations[index]

    def __len__(self):
        return len(self.image_ids)

def batch_collate(batch, visual_mode):

    # Stack the image tensors, doing padding if necessary for the sequence of region features
    image_tensors = [x[0] for x in batch]
    if visual_mode == 'pil-image':
        images = image_tensors                                          # Not actually tensors for this option, list of PIL.Image objects
    if visual_mode == 'raw':
        images = torch.stack(image_tensors, dim=0)               # Stacks individual raw image tensors to give (B, 3, W, H) tensor
    elif visual_mode == 'fast-rcnn':
        max_len = max([t.shape[0] for t in image_tensors])
        image_tensors_padded = []
        for i in range(len(image_tensors)):
            padding_tensor = torch.zeros(max_len-image_tensors[i].shape[0], image_tensors[i].shape[1])
            padded_tensor = torch.cat((image_tensors[i], padding_tensor), dim=0)
            assert padded_tensor.shape[0] == max_len
            image_tensors_padded.append(padded_tensor)
        images = torch.stack(image_tensors_padded, dim=0)        # Pads region features with 0 vectors to give (B, R, hv) tensor
    
    targets = torch.stack([x[1] for x in batch], dim=0)

    return {'images': images,
            'targets': targets}



def build_mscoco_detection_dataloader(args, images_dir, annotation_dir, split, visual_mode):
    ###Dataloader for MSCOCO detection
    logger.info("Creating MSCOCO-Detection {} dataloader with batch size of {}".format(split, args.batch_size))

    #dataset    = root, annFile, transform=None, target_transform=None):
    mscoco_images_dataset = MSCOCOImagesDataset(images_dir)
    dataset    = MSCOCODetection(annotation_dir=annotation_dir,
                                    split=split,
                                    images_dataset = mscoco_images_dataset,
                                    visual_mode=visual_mode)
    shuffle = True if split == 'train' else False

    dataloader = torch.utils.data.DataLoader(dataset,
                                            num_workers=args.num_workers,
                                            batch_size=args.batch_size,
                                            shuffle=shuffle,
                                            collate_fn=lambda x: batch_collate(x, visual_mode))
    return dataloader

if __name__ == '__main__':

    class Args:
        def __init__(self):
            self.batch_size = 4
            self.num_workers = 2
            self.visual_mode = 'raw'

    args = Args()
    images_dir          = '/data/datasets/MCL/ms-coco/'
    annotation_dir     = '/data/datasets/MCL/ms-coco/detections/annotations/'
    split               = 'train'
    annotation_file     = os.path.join(annotation_dir, 'instances_{}2017.json'.format(split))

    mscoco_images_dataset = MSCOCOImagesDataset(images_dir)
    print(mscoco_images_dataset.get_image_data( 1, 'raw').shape)
    print(mscoco_images_dataset.get_image_data( 9, 'raw').shape)

    mscoco_detection_train_dataloader = build_mscoco_detection_dataloader(args, images_dir, annotation_dir, split='train', visual_mode=args.visual_mode)
    mscoco_detection_val_dataloader = build_mscoco_detection_dataloader(args, images_dir, annotation_dir, split='val', visual_mode=args.visual_mode)

