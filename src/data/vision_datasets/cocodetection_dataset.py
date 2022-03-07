import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import logging
import pdb

## image_dataset -- tejas
from data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class MSCOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, image_datasets):
        self.root = root
        self.image_datasets = image_datasets
        self.transforms = self.image_datasets.transform
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        images = []
        annotations = []
        print("loading information into memory...")
        ###load all the images and annotations
        #for index in self.ids:
        for index in self.ids:
            # Own coco file
            coco = self.coco
            # Image ID
            img_id = index #self.ids[index]
            #print(img_id)
            #img = Image.open(os.path.join(self.root, path))    # open the input image
            #img   = self.image_datasets.get_image_data(img_id, 'patch') 
            img   = self.image_datasets.get_image_data(img_id, 'raw')
            
            #print(img.shape)
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
            
            ###This add the binary mask to annotations 
            bmask = np.zeros((img_info['height'],img_info['width']))
            for i in range(len(anns)):
                bmask = np.maximum(self.coco.annToMask(anns[i]), bmask)
            #print('bmask.shape: ', bmask.shape)
            #From numpy to PIL Image to tensor(transform)
            my_annotation["bmask"] = self.transforms(Image.fromarray(np.uint32(bmask), 'RGB'))

            ###This add the multiclass mask
            mask = np.zeros((img_info['height'], img_info['width']))
            for i in range(len(anns)):
                #print(anns[i]['category_id'], len(cats))
                className = self.getClassName(anns[i]['category_id'], cats)
                pixel_value = cat_names.index(className)+1
                mask = np.maximum(self.coco.annToMask(anns[i])*pixel_value, mask)
            #print('mask.shape: ', mask.shape)
            #From numpy to PIL Image to tensor(transform)
            my_annotation["mask"] = self.transforms(Image.fromarray(np.uint32(mask), 'RGB'))
            #print(img.shape, my_annotation['bmask'].shape)
            images.append(img)
            annotations.append(my_annotation)
        self.images      = images
        self.annotations = annotations
        print("I finished loading all the annotations into memory")

    def getClassName(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"
        #print('The class name is', getClassName(77, cats))

class MSCOCOSegmentation(MSCOCODataset):
    def __init__(self, root, annotation, image_datasets):
        super(MSCOCOSegmentation, self).__init__(root, annotation, image_datasets)

    def __getitem__(self, index):
        #return self.image_datasets.get_image_data(index, 'raw'), self.annotations[index]['bmask']
        return self.images[index], self.annotations[index]['bmask']
    def __len__(self):
        return len(self.images)
        

class MSCOCODetection(MSCOCODataset):
    def __init__(self, root, annotation, image_datasets):
        super(MSCOCODetection, self).__init__(root, annotation, image_datasets)

    def __getitem__(self, index):
        #return self.image_datasets.get_image_data(index, 'raw'), self.annotations[index]['boxes']
        return self.images[index], self.annotations[index]['boxes']
    def __len__(self):
        return len(self.images)

class Args:
    def __init__(self):
        self.batch_size = 4
        self.shuffle = True
        self.num_workers = 2

def build_mscoco_dataloader(args, dataset, split):
    ###Dataloader for MSCOCO
    shuffle = True if split == 'train' else False
    logger.info("Creating MSCOCO  {} dataloader with batch size of {}".format(split, args.batch_size))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=shuffle)
    return dataloader


if __name__ == '__main__':

    mscoco_image_datasets = MSCOCOImagesDataset('/data/datasets/MCL/ms-coco/')
    print(mscoco_image_datasets.get_image_data( 1, 'raw').shape)
    print(mscoco_image_datasets.get_image_data( 9, 'raw').shape)

    images_dir          = '/data/datasets/MCL/ms-coco/'
    annotation_dir     = '/data/datasets/MCL/ms-coco/detections/annotations/'
    split               = 'train'
    annotation_file     = os.path.join(annotation_dir, 'instances_{}2017.json'.format(split))
    
    trainset    = MSCOCOSegmentation(root=images_dir+ 'images/',
                                            annotation=annotation_file,
                                            image_datasets = MSCOCOImagesDataset(images_dir),
                                            #transforms=get_transform()
                                            )

    img, bmask  = trainset[0]
    print(img.shape, bmask.shape)

    args = Args()
    mscoco_dataloader_train = build_mscoco_dataloader(args, trainset, split='train')
    mscoco_dataloader_val = build_mscoco_dataloader(args, trainset, split='val')

