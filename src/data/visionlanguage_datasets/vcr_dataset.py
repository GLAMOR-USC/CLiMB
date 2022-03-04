import sys
import os
import json
import logging
import random
import glob
from tqdm import tqdm
from collections import defaultdict
import pdb

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
from copy import deepcopy

from utils.box_utils import load_image, resize_image, to_tensor_and_normalize
from utils.mask_utils import make_mask

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)


GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']


class VCRDataset(Dataset):
    def __init__(self, data_dir, split, mode, tokenizer, only_use_relevant_dets=True, add_image_as_a_box=False):
        """

        :param split: train, val, or test
        :param mode: answer or rationale
        :param only_use_relevant_dets: True, if we will only use the detections mentioned in the question and answer.
                                       False, if we should use all detections.
        :param add_image_as_a_box:     True to add the image in as an additional 'detection'. It'll go first in the list
                                       of objects. #TODO
        """
        self._check_valid(split, mode)
        self.data_dir = data_dir
        self.split = split
        self.mode = mode
        self.only_use_relevant_dets = only_use_relevant_dets
        print("Only relevant dets" if only_use_relevant_dets else "Using all detections", flush=True)

        img_dir = os.path.join(data_dir, 'vcr1images')
        # object to index
        with open(os.path.join(data_dir, 'cocoontology.json'), 'r') as f:
            coco = json.load(f)
        self.coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)} #{'__background__': 0, 'person': 1, 'bicycle': 2, 'car': 3}

        self.items = []
        with open(os.path.join(data_dir, '{}.jsonl'.format(split)), 'r') as f:
            for s in tqdm(f):
                item = json.loads(s) 

                ctx = item['question'] # ['How', 'is', [0], 'feeling', '?']
                if mode == 'rationale':
                    label = item['rationale_label']
                    ctx += item['answer_choices'][item['answer_label']] # condition on the correct answer
                else:
                    label = item['answer_label']

                item['ctx'] = ctx
                dets2use, old_det_to_new_ind = self._get_dets_to_use(item)


                '''
                new_item = {
                    'img_fn': item['img_fn'],
                    'ctx_ids': ,
                    'ctx_tags': ,
                    'ans_ids':,
                    'ans_tags':,
                    'label': label,
                    'dets2use' = dets2use,
                    'old_det_to_new_ind' = old_det_to_new_ind
                }
                self.items.append(new_items)
                '''


    def _get_dets_to_use(self, item):
        """
        Use fewer object detections & re-index the object
        """
        # Load questions and answers
        ctx = item['ctx']
        answer_choices = item['{}_choices'.format(self.mode)]

        if self.only_use_relevant_dets:
            dets2use = np.zeros(len(item['objects']), dtype=bool)
            people = np.array([x == 'person' for x in item['objects']], dtype=bool)
            for sent in answer_choices + [ctx]:
                for possibly_det_list in sent:
                    if isinstance(possibly_det_list, list):
                        for tag in possibly_det_list:
                            if tag >= 0 and tag < len(item['objects']):  # sanity check
                                dets2use[tag] = True
                    elif possibly_det_list.lower() in ('everyone', 'everyones'):
                        dets2use |= people
            if not dets2use.any():
                dets2use |= people
        else:
            dets2use = np.ones(len(item['objects']), dtype=bool)

        # we will use these detections
        dets2use = np.where(dets2use)[0]

        old_det_to_new_ind = np.zeros(len(item['objects']), dtype=np.int32) - 1
        old_det_to_new_ind[dets2use] = np.arange(dets2use.shape[0], dtype=np.int32)

        old_det_to_new_ind = old_det_to_new_ind.tolist()
        return dets2use, old_det_to_new_ind


    def _check_valid(self, split, mode):
        if split not in ('test', 'train', 'val'):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format(mode))

        if mode not in ('answer', 'rationale'):
            raise ValueError("split must be answer or rationale")

    def __getitem__(self, index):
        item = deepcopy(self.items[index])
        ###################################################################
        # Load image now and rescale it. Might have to subtract the mean and whatnot here too.
        image = load_image(os.path.join(img_dir, item['img_fn']))
        image, window, img_scale, padding = resize_image(image, random_pad=(split=='train'))
        image = to_tensor_and_normalize(image) #[c, h, w]
        ###################################################################
        # Load boxes.
        with open(os.path.join(img_dir, item['metadata_fn']), 'r') as f:
            metadata = json.load(f)

        # [nobj, 14, 14]
        segms = np.stack([make_mask(mask_size=14, box=metadata['boxes'][i], polygons_list=metadata['segms'][i])
                          for i in dets2use])

        # Chop off the final dimension, which is the confidence
        boxes = np.array(metadata['boxes'])[dets2use, :-1]
        boxes *= img_scale
        boxes[:, :2] += np.array(padding[:2])[None] # padding[:2] = (left_pad, top_pad)
        boxes[:, 2:] += np.array(padding[:2])[None]
        obj_labels = [self.coco_obj_to_ind[item['objects'][i]] for i in dets2use.tolist()]
        try:
            assert np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2]))
        except:
            pdb.set_trace()


def build_vcr_dataloader(args, tokenizer):
    dataset = VCRDataset(args.data_dir, args.split, args.mode, tokenizer)

    '''
    num_labels = dataset.num_labels
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=(args.split == 'train'),
        collate_fn=lambda x: batch_collate(x, tokenizer, args, num_labels))
    return dataloader
    '''


if __name__ == '__main__':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    data_dir = '/data/datasets/MCL/vcr/'
    class Args:
        def __init__(self):
            self.batch_size = 4
            self.num_workers = 2
            self.data_dir = data_dir
            self.split = 'val'
            self.mode = 'answer'
    args = Args()

    vcr_dataloader = build_vcr_dataloader(args, tokenizer)
