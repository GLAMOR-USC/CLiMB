import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import numpy as np


class cocoSegmentation(torch.utils.data.Dataset):
    def __init__(self, root, annotation, image_datasets, transforms=None):
        super(cocoSegmentation, self).__init__()
        self.root = root
        self.transforms = transforms
        self.image_datasets = image_datasets
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def getClassName(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"
        #print('The class name is', getClassName(77, cats))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        ##img   = self.image_dataset.get_image_data(img_id, 'patch') 
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
        # open the input image
        img = Image.open(os.path.join(self.root, path))

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
        bmask = torch.zeros((img_info['height'],img_info['width']))
        for i in range(len(anns)):
            bmask = np.maximum(self.coco.annToMask(anns[i]), bmask)
        my_annotation["bmask"] = bmask

        ###This add the multiclass mask
        mask = torch.zeros((img_info['height'], img_info['width']))
        for i in range(len(anns)):
            #print(anns[i]['category_id'], len(cats))
            className = self.getClassName(anns[i]['category_id'], cats)
            pixel_value = cat_names.index(className)+1
            mask = np.maximum(self.coco.annToMask(anns[i])*pixel_value, mask)
        my_annotation["mask"] = mask

        ##Tejas function
        if self.transforms is not None:
            img = self.transforms(img)
        ##img   = self.image_dataset.get_image_data(img_id, 'patch')
        
        return img, my_annotation

    def __len__(self):
        return len(self.ids)