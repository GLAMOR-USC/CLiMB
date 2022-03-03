import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
## image_dataset -- tejas
from data.image_datasets import cocoimages_dataset


class MSCOCOSegmentation(torch.utils.data.Dataset):
    def __init__(self, root, annotation, image_datasets):
        super(MSCOCOSegmentation, self).__init__()
        self.root = root
        self.image_datasets = image_datasets
        self.transforms = self.image_datasets.transform
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
        #print(img_id)
        #img = Image.open(os.path.join(self.root, path))    # open the input image
        img   = self.image_datasets.get_image_data(img_id, 'patch') 
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
        return img, my_annotation

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':

    tejas = MSCOCOImagesDataset('/data/datasets/MCL/ms-coco/')
    print(tejas.get_image_data( 9, 'patch').shape)

    images_dir          = '/data/datasets/MCL/ms-coco/'
    annotations         = '/data/datasets/MCL/ms-coco/detections/annotations/instances_train2017.json'
    trainset            = MSCOCOSegmentation(root=images_dir+ 'images/',
                                            annotation=annotations,
                                            image_datasets = MSCOCOImagesDataset(images_dir),
                                            #transforms=get_transform()
                        )

    img, annotations = trainset[0]
    print(img.shape)
    print(annotations['bmask'].shape)
    print(annotations['mask'].shape)

