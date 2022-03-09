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

from torchvision import transforms as T

## image_dataset -- tejas
from data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class MSCOCOSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_dir, split, images_dataset, visual_mode, mask_type):

        self.images_dataset = images_dataset
        self.transform = T.Compose([
            T.Resize(images_dataset.image_size, interpolation=Image.NEAREST),
            T.ToTensor(),                                 # [0, 1]
        ])
        self.visual_mode = visual_mode
        self.mask_type = mask_type
        assert self.mask_type in ['binary', 'semantic']

        annotation_file     = os.path.join(annotation_dir, 'instances_{}2017.json'.format(split))
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        coco_cached_file = os.path.join(annotation_dir, 'cached_data', 'coco_segmentation_{}_{}.pkl'.format(mask_type, split))

        if os.path.isfile(coco_cached_file):
            # Load image IDs and annotations from cached file
            data = pkl.load(open(coco_cached_file, 'rb'))
            self.annotations = data['annotations']
            logger.info("Finished loading all the COCO annotations into memory")

        else:
            annotations = []
            logger.info("Loading COCO information into memory...")
            ###load all the images and annotations
            #for index in self.ids:
            for index in tqdm(self.ids):
                # Own coco file
                coco = self.coco
                # Image ID
                img_id = index #self.ids[index]
                
                # List: get annotation id from coco
                ann_ids = coco.getAnnIds(imgIds=img_id)
                img_info = self.coco.loadImgs([img_id])[0]
                #Get all the annotations for the specified image.
                ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
                anns = self.coco.loadAnns(ann_ids)

                # Category IDs.
                cat_ids = self.coco.getCatIds()
                # All categories.
                cats = self.coco.loadCats(cat_ids)
                cat_names = [cat["name"] for cat in cats]

                annotation = {'image_id': img_id,
                        'anns': anns,
                        'img_info': img_info,
                        'cats': cats,
                        'cat_names': cat_names}

                annotations.append(annotation)

            self.annotations = annotations
            logger.info("Finished loading all the COCO annotations into memory")

            # Cache the annotations
            cache_data = {
                        'annotations': self.annotations
                        }
            pkl.dump(cache_data, open(coco_cached_file, 'wb'))

        logger.info("Loaded COCO Segmentation {} set with {} examples".format(split, len(self.annotations)))

    def getClassName(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"
        #print('The class name is', getClassName(77, cats))

    def __getitem__(self, index):

        anno = self.annotations[index]
        image_id = anno['image_id']
        image = self.images_dataset.get_image_data(image_id, self.visual_mode)

        anns = anno['anns']
        img_info = anno['img_info']
        if self.mask_type == 'binary':
            ###This add the binary mask to annotations
            bmask = np.zeros((img_info['height'],img_info['width']))
            for i in range(len(anns)):
                bmask = np.maximum(self.coco.annToMask(anns[i]), bmask)
            mask = bmask
        elif self.mask_type == 'semantic':
            ###This add the multiclass mask
            cats = anno['cats']
            cat_names = anno['cat_names']
            mask = np.zeros((img_info['height'], img_info['width']))
            for i in range(len(anns)):
                #print(anns[i]['category_id'], len(cats))
                className = self.getClassName(anns[i]['category_id'], cats)
                pixel_value = cat_names.index(className)+1
                mask = np.maximum(self.coco.annToMask(anns[i])*pixel_value, mask)
        mask = self.transform(Image.fromarray(mask))
        return image, mask

    def __len__(self):
        return len(self.annotations)


def build_mscoco_segmentation_dataloader(args, annotation_dir, split, visual_mode, mask_type):
    ###Dataloader for MSCOCO Segmentation
    logger.info("Creating MSCOCO-Segmentation {} dataloader with batch size of {}".format(split, args.batch_size))
    dataset    = MSCOCOSegmentationDataset(annotation_dir=annotation_dir,
                                           split=split,
                                           images_dataset = mscoco_images_dataset,
                                           visual_mode=visual_mode,
                                           mask_type=mask_type
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
            self.mask_type = 'binary'

    args = Args()
    images_dir          = '/data/datasets/MCL/ms-coco/'
    annotation_dir     = '/data/datasets/MCL/ms-coco/detections/annotations/'
    split               = 'train'
    annotation_file     = os.path.join(annotation_dir, 'instances_{}2017.json'.format(split))

    mscoco_images_dataset = MSCOCOImagesDataset(images_dir)
    print(mscoco_images_dataset.get_image_data( 1, 'raw').shape)
    print(mscoco_images_dataset.get_image_data( 9, 'raw').shape)
    segmentation_train_set    = MSCOCOSegmentationDataset(annotation_dir=annotation_dir,
                                                        split='train',
                                                        images_dataset = mscoco_images_dataset,
                                                        visual_mode=args.visual_mode,
                                                        mask_type=args.mask_type
                                                        )

    img, bmask  = segmentation_train_set[0]
    print(img.shape, bmask.shape)

    mscoco_segmentation_train_dataloader = build_mscoco_segmentation_dataloader(args, annotation_dir, split='train', visual_mode=args.visual_mode, mask_type=args.mask_type)
    mscoco_segmentation_val_dataloader = build_mscoco_segmentation_dataloader(args, annotation_dir, split='val', visual_mode=args.visual_mode, mask_type=args.mask_type)
    pdb.set_trace()
