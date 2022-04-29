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
                    temporal_text =  'the ' + str(lst_colors[int(subelement)]).strip() #+ ' ' + str(objects[int(subelement)])
                else:
                    temporal_text = 'the gray ' + str(objects[int(subelement)]).strip()
        elif(type(element) == int):
            if(objects[int(element)] == 'person'):
                temporal_text = 'the ' + str(lst_colors[int(element)]).strip() #+ ' ' + str(objects[int(element)])
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
                
                image_id = os.path.join(data_dir, 'drawn_images/bbox/' + str(split) + '/' + str(task_type)+ '/' + str(line['annot_id']) +'.jpg')  ## train-0, train-1, train-2
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

                doc = {'image_id': image_id,
                        'text': texts,
                        #'text_input_ids': input_ids,
                        'label': label}
                self.data.append(doc)
                
            pkl.dump(self.data, open(self.cached_data_file, 'wb'))
        self.n_examples = len(self.data)
        logger.info("Loaded VCR {} dataset, with {} examples".format(self.split, len(self.data)))

    def __len__(self):
        return self.n_examples

    def __getitem__(self, index):
        example = self.data[index]
        
        image_fn     = example['image_id']
        pil_transform = T.Resize(size=384, max_size=640)
        image = Image.open(image_fn)
        image = image.convert('RGB')
        if min(list(image.size)) > 384:
            image = pil_transform(image)

        text    = example['text']
        label   = example['label']

        return  text, image, label

def vcr_batch_collate(batch):
    
    texts, pil_objs, labels = zip(*batch)
    return {'texts': list(texts), 
            'images': pil_objs, 
            'labels': torch.LongTensor(labels)}

def build_vcr_dataloader(args, data_dir, split, tokenizer, task_type):
    
    batch_size = args.batch_size
    shuffle = True if split == 'train' else False

    logger.info("Creating vcr {} dataloader with batch size of {}".format(split, batch_size))

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
        collate_fn=lambda x: vcr_batch_collate(x))
    return dataloader


    #### let's concatenate:
    
if __name__ == '__main__':

    class Args:
        def __init__(self):
            self.batch_size = 4
            self.num_workers = 2

    class Args:
        def __init__(self):
            self.batch_size = 4
            self.num_workers = 2

    args = Args()
    data_dir          = '/data/datasets/MCL/vcr/'
    #annotation_dir    = '/data/datasets/MCL/vcr/annotation/'
    split             = 'val' #'train' 
    text    = ['Why', 'is', [0], 'smiling', 'at', [1], '?']
    objects = ['person', 'person', 'bottle']
    #process_list(text, objects)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #vcr.VCRDataset(data_dir, split, tokenizer, task_type='answer')

    vcr_train_dataloader  = vcr.build_vcr_dataloader(args, data_dir, split= 'train', tokenizer = tokenizer, task_type = 'answer')
    vcr_train_dataloader  = vcr.build_vcr_dataloader(args, data_dir, split= 'val',  tokenizer = tokenizer, task_type = 'answer')

    print(len(vcr_train_dataloader))

    vcr_train_dataloader  = vcr.build_vcr_dataloader(args, data_dir, split= 'train', tokenizer = tokenizer, task_type = 'rationale')
    vcr_train_dataloader  = vcr.build_vcr_dataloader(args, data_dir, split= 'val',  tokenizer = tokenizer, task_type = 'rationale')

    print(len(vcr_train_dataloader))

    for batch in tqdm(vcr_train_dataloader):
            print(batch['texts'])
            print(batch['images'])
            print(batch['labels'])
            pdb.set_trace() 


#main()