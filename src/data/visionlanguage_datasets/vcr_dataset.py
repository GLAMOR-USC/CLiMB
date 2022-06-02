import sys
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
import pdb
import jsonlines

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset


from transformers import BertTokenizer

from PIL import Image
#from utils.image_utils import resize_image


logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Skyler', 'Frankie', 'Pat', 'Quinn', 'Morgan', 'Finley', 'Harley', 'Robbie', 'Sidney', 'Tommie',
                        'Ashley', 'Carter', 'Adrian', 'Clarke', 'Logan', 'Mickey', 'Nicky', 'Parker', 'Tyler',
                        'Reese', 'Charlie', 'Austin', 'Denver', 'Emerson', 'Tatum', 'Dallas', 'Haven', 'Jordan',
                        'Robin', 'Rory', 'Bellamy', 'Salem', 'Sutton', 'Gray', 'Shae', 'Kyle', 'Alex', 'Ryan',
                        'Cameron', 'Dakota']


def process_list(mytext, objects):
    ## Read file with the name of the color per object
    #file = open('colors.txt', 'r')  #lst_colors = file.readlines() #file.close()
    lst_colors = ['person'+ str(i) for i in range(1,57)]
    #print(lst_colors)
    
    ## processing the text
    text = ''
    for element in mytext:
        #print(element)
        if(type(element) == list):     #### If it's a list we need to process each object 
            for subelement in element:
                if(objects[int(subelement)] == 'person'):
                    #temporal_text =  'the ' + str(lst_colors[int(subelement)]).strip() #+ ' ' + str(objects[int(subelement)])
                    temporal_text = GENDER_NEUTRAL_NAMES[int(subelement)]
                else:
                    temporal_text = 'the gray ' + str(objects[int(subelement)]).strip()
        elif(type(element) == int):
            if(objects[int(element)] == 'person'):
                #temporal_text = 'the ' + str(lst_colors[int(element)]).strip() #+ ' ' + str(objects[int(element)])
                temporal_text = GENDER_NEUTRAL_NAMES[int(subelement)]
            else:
                temporal_text = 'the gray ' + str(objects[int(subelement)])
        else:
            temporal_text = element
        text += temporal_text + ' '
    #print('text: ', text)
    return text

class VCRDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, task_type='answer'):

        self.data_dir = data_dir
        self.images_dataset = self.data_dir + 'draw_images/bbox/'    
            
        #self.images_dataset = images_dataset
        self.image_dir = os.path.join(data_dir, 'vcr')
        self.split = split
        self.tokenizer = tokenizer
        self.task_type = task_type
        #self.visual_mode = visual_mode

        self.annotations_file = os.path.join(data_dir, 'annotation/{}.jsonl'.format(split))
        #self.categories = ['entailment', 'contradiction', 'neutral']
        #self.cat2label = {cat: i for i, cat in enumerate(self.categories)}

        self.cached_data_file = os.path.join(data_dir, 'cached_vcr_data', 'vcr_'+ str(task_type) + '_' + '{}.pkl'.format(split))
        if os.path.isfile(self.cached_data_file):
            self.data = pkl.load(open(self.cached_data_file, 'rb'))
        else:
            self.data = []
            json_lines = jsonlines.open(self.annotations_file)
            count = 0
            for line in json_lines:
                
                image_path = os.path.join('drawn_images/bbox/' + str(split) + '/' + str(task_type)+ '/' + str(line['annot_id']) +'.jpg')  ## train-0, train-1, train-2
                #print(image_id)
                #exit()
                texts = []
                #print('task_type: ', task_type)
                objects = line['objects']   ### objects
                
                question = process_list(line['question'], objects)  ### question
                if(task_type == 'answer'):
                    ### answers:     question + ' [SEP] ' + answer
                    for answer in line['answer_choices']:                        
                        answer1 = process_list(answer, objects)
                        text = question + ' [SEP] ' + answer1
                        #print('answer: ', text)
                        texts.append(text)
                    label = int(line['answer_label']) ##number
                else:
                    ### rationales:  question + '[SEP]' + answer + '[SEP]' + rationale
                    answer  = process_list( line['answer_choices'][int(line['answer_label'])], objects)
                    for rationale in line['rationale_choices']:
                        rationale1 = process_list(rationale, objects)
                        text = question + ' [SEP] ' + answer + ' [SEP] ' + rationale1
                        #print('rationale: ', text)
                        texts.append(text)
                    
                    label = int(line['rationale_label']) ##number
                #print('image_id: ', image_id,'\n, text: ', texts, '\n, label: ', label)

                #tokens = [self.tokenizer.tokenize(text) for text in texts]
                #print(tokens)
                #####input_ids = self.tokenizer.convert_tokens_to_ids(tokens[0])

                doc = {'image_path': image_path,
                        'text': texts,
                        #'text_input_ids': input_ids,
                        'label': label}
                self.data.append(doc)
                
            pkl.dump(self.data, open(self.cached_data_file, 'wb'))
        self.n_examples = len(self.data)
        logger.info("Loaded VCR-{} {} dataset, with {} examples".format(self.task_type, self.split, len(self.data)))

    def __len__(self):
        return self.n_examples

    def __getitem__(self, index):
        example = self.data[index]
        
        image_fn     = os.path.join(self.data_dir, example['image_path'])
        pil_transform = T.Resize(size=384, max_size=640)
        image = Image.open(image_fn)
        image = image.convert('RGB')
        if min(list(image.size)) > 384:
            image = pil_transform(image)

        text    = example['text']
        label   = example['label']

        return  text, image, label

    def convert_to_low_shot(self, low_shot_percentage):

        assert self.split == 'train'
        logger.info("Converting VCR train split into low-shot dataset, with {:.2f}% training samples...".format(low_shot_percentage*100.0))
        n_low_shot_examples = int(low_shot_percentage*self.n_examples)

        new_data = random.sample(self.data, n_low_shot_examples)
        self.data = new_data
        self.n_examples = len(self.data)

        logger.info("Converted into low-shot dataset, with {} examples".format(self.n_examples))

def vcr_batch_collate(batch, visual_mode):
    
    if visual_mode == 'pil-image':
        texts, pil_objs, labels = zip(*batch)

    return {'raw_texts': list(texts),
            'images': list(pil_objs),
            'labels': torch.LongTensor(labels)}

def build_vcr_dataloader(args, data_dir, split, tokenizer, task_type, visual_mode):
    
    batch_size = int(args.batch_size/4)
    shuffle = True if split == 'train' else False

    logger.info("Creating VCR {} dataloader with batch size of {}".format(split, batch_size))

    dataset = VCRDataset(data_dir, split, tokenizer, task_type)
    
    #image, text, label = dataset[0]
    #print(type(image))
    #print(type(text), text)
    #print(type(label), label)

    #num_labels = dataset.num_labels
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: vcr_batch_collate(x, visual_mode))
    return dataloader


    #### let's concatenate:
    
if __name__ == '__main__':

    class Args:
        def __init__(self):
            self.batch_size = 4
            self.num_workers = 2
            self.visual_mode = 'pil-image'
    
    args = Args()
    data_dir          = '/data/datasets/MCL/vcr/'
    #annotation_dir    = '/data/datasets/MCL/vcr/annotation/'
    split             = 'val' #'train' 
    text    = ['Why', 'is', [0], 'smiling', 'at', [1], '?']
    objects = ['person', 'person', 'bottle']
    #process_list(text, objects)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #vcr.VCRDataset(data_dir, split, tokenizer, task_type='answer')

    vcr_train_dataloader  = build_vcr_dataloader(args, data_dir, split= 'train', tokenizer = tokenizer, task_type = 'answer', visual_mode=args.visual_mode)
    vcr_val_dataloader  = build_vcr_dataloader(args, data_dir, split= 'val',  tokenizer = tokenizer, task_type = 'answer', visual_mode=args.visual_mode)

    vcr_train_dataloader  = build_vcr_dataloader(args, data_dir, split= 'train', tokenizer = tokenizer, task_type = 'rationale', visual_mode=args.visual_mode)
    vcr_val_dataloader  = build_vcr_dataloader(args, data_dir, split= 'val',  tokenizer = tokenizer, task_type = 'rationale', visual_mode=args.visual_mode)

    #max_token_len = 0
    #len_over_40 = 0
    #total = 0
    #for batch in tqdm(vcr_val_dataloader):
    #        #print(batch['texts'])
    #        #print(batch['images'])
    #        #print(batch['labels'])
    #        for answer_set in batch['texts']:
    #            for choice in answer_set:
    #                tokens = tokenizer.tokenize(choice)
    #                ids = tokenizer.convert_tokens_to_ids(tokens)
    #                if len(ids) >= 40:
    #                    len_over_40 += 1
    #                max_token_len = max(max_token_len, len(ids))
    #                total += 1
    #        #pdb.set_trace() 
    #print("Percent of samples with > 40 token len: {:.4f}%".format(100.0*len_over_40/total))
    #print("Maximum length = {}".format(max_token_len))


#main()