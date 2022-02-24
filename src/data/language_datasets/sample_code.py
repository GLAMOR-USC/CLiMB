from pprint import pprint
from tqdm import tqdm
import pdb
import logging
import os
import pickle
import numpy as np
from transformers import BertTokenizer

from data.language_datasets.text_processors import *
from data.language_datasets.text_dataset import *


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    cache_dir = "/data/datasets/MCL/cached_datasets"
    dataloader = get_data_loader(tokenizer, task_name='sst2', split='train', max_len=100, batch_size=128, cache_dir=cache_dir) 
    dataloader = get_data_loader(tokenizer, task_name='imdb', split='dev', max_len=100, batch_size=128, cache_dir=cache_dir) 

    dataloader = get_data_loader(tokenizer, task_name='hellaswag', split='test', max_len=100, batch_size=128, 
        cache_dir=cache_dir, data_dir='/data/datasets/MCL/hellaswag') 
    dataloader = get_data_loader(tokenizer, task_name='piqa', split='train', max_len=100, batch_size=128, 
        cache_dir=cache_dir, data_dir='/data/datasets/MCL/piqa') 
    dataloader = get_data_loader(tokenizer, task_name='cosmosqa', split='test', max_len=150, batch_size=128, 
        cache_dir=cache_dir, data_dir='/data/datasets/MCL/cosmosqa') 

    for (input_ids, attn_masks, token_type_ids, labels) in dataloader:
        if len(input_ids.shape)>2:
            print(tokenizer.batch_decode(input_ids[0]), input_ids.shape)
        else:
            print(tokenizer.decode(input_ids[0]), input_ids.shape)
        print(attn_masks[0], attn_masks.shape)
        print(token_type_ids[0], token_type_ids.shape)
        print(labels[0], labels.shape)
        pdb.set_trace()
