import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import logging
import pdb
from tqdm import tqdm
import pickle as pkl

## image_dataset -- tejas
from data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class MSCOCODetectionDataset(torch.utils.data.Dataset):

    def __init__(self, annotation_dir, split, images_dataset, visual_mode):
        
        self.images_dataset = images_dataset
        self.transforms = self.images_dataset.raw_transform
        self.visual_mode = visual_mode

        annotation_file     = os.path.join(annotation_dir, 'instances_{}2017.json'.format(split))
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        coco_cached_file = os.path.join(annotation_dir, 'cached_data', 'coco_detection_{}.pkl'.format(split))

        if os.path.isfile(coco_cached_file):
            # Load image IDs and annotations from cached file
            data = pkl.load(open(coco_cached_file, 'rb'))
            self.image_ids = data['image_ids']
            self.annotations = data['annotations']
            logger.info("Finished loading all the COCO annotations into memory")

        else:
            image_ids = []
            annotations = []
            logger.info("Loading COCO information into memory...")
            ###load all the images and annotations
            #for index in self.ids:
            for index in self.ids:
                # Own coco file
                coco = self.coco
                # Image ID
                img_id = index #self.ids[index]
                
                # List: get annotation id from coco
                ann_ids = coco.getAnnIds(imgIds=img_id)
                # Dictionary: target coco_annotation file for an image
                coco_annotation = coco.loadAnns(ann_ids)
                img_info = self.coco.loadImgs([img_id])[0]
                #Get all the annotations for the specified image.
                ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
                anns = self.coco.loadAnns(ann_ids)

                # Category IDs.
                cat_ids = self.coco.getCatIds()
                # All categories.
                cats = self.coco.loadCats(cat_ids)
                cat_names = [cat["name"] for cat in cats]

                # path for input image
                path = coco.loadImgs(img_id)[0]['file_name']
                
                # number of objects in the image
                num_objs = len(coco_annotation)

                # Bounding boxes for objects
                # In coco format, bbox = [xmin, ymin, width, height]
                # In pytorch, the input should be [xmin, ymin, xmax, ymax]
                boxes = []
                for i in range(num_objs):
                    xmin = coco_annotation[i]['bbox'][0]
                    ymin = coco_annotation[i]['bbox'][1]
                    xmax = xmin + coco_annotation[i]['bbox'][2]
                    ymax = ymin + coco_annotation[i]['bbox'][3]
                    boxes.append([xmin, ymin, xmax, ymax])
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                # Labels (In my case, I only one class: target class or background)
                labels = torch.ones((num_objs,), dtype=torch.int64)
                # Tensorise img_id
                img_id = torch.tensor([img_id])
                # Size of bbox (Rectangular)
                areas = []
                for i in range(num_objs):
                    areas.append(coco_annotation[i]['area'])
                areas = torch.as_tensor(areas, dtype=torch.float32)
                # Iscrowd
                iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

                # Annotation is in dictionary format
                my_annotation = {}
                my_annotation["boxes"] = boxes
                my_annotation["labels"] = labels
                my_annotation["image_id"] = img_id
                my_annotation["area"] = areas
                my_annotation["iscrowd"] = iscrowd

                image_ids.append(img_id)
                annotations.append(my_annotation)

            self.image_ids      = image_ids
            self.annotations = annotations
            logger.info("Finished loading all the COCO annotations into memory")

            # Cache the annotations
            cache_data = {
                        'image_ids': self.image_ids,
                        'annotations': self.annotations
                        }
            pkl.dump(cache_data, open(coco_cached_file, 'wb'))

        logger.info("Loaded COCO detection {} set with {} examples".format(split, len(self.image_ids)))

    def getClassName(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"
        #print('The class name is', getClassName(77, cats))

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = self.images_dataset.get_image_data(img_id, self.visual_mode)
        return image, self.annotations[index]['boxes']

    def __len__(self):
        return len(self.image_ids)

def build_mscoco_detection_dataloader(args, annotation_dir, split, visual_mode):
    ###Dataloader for MSCOCO detection
    logger.info("Creating MSCOCO-Detection {} dataloader with batch size of {}".format(split, args.batch_size))
    dataset    = MSCOCODetectionDataset(annotation_dir=annotation_dir,
                                    split=split,
                                    images_dataset = mscoco_images_dataset,
                                    visual_mode=visual_mode
                                    )
    shuffle = True if split == 'train' else False

    dataloader = torch.utils.data.DataLoader(dataset,
                                            num_workers=args.num_workers,
                                            batch_size=args.batch_size,
                                            shuffle=shuffle)
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

    mscoco_detection_train_dataloader = build_mscoco_detection_dataloader(args, annotation_dir, split='train', visual_mode=args.visual_mode)
    mscoco_detection_val_dataloader = build_mscoco_detection_dataloader(args, annotation_dir, split='val', visual_mode=args.visual_mode)

