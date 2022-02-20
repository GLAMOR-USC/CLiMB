from tqdm import tqdm
import pdb
import logging
import os
import pickle
import numpy as np
from transformers import PreTrainedTokenizer
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from text_processors import *

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def convert_mc_data_to_features(
    dataset,
    tokenizer: PreTrainedTokenizer,
    task_name,
    max_len_clip,
):
    max_seq_len = 0
    features, all_lengths = [], []
    for (ex_index, example) in tqdm(enumerate(dataset), desc="convert examples to features"):
        choice_feats = []
        if task_name == 'cosmosqa': #TODO: clip the length of example['text_a']
            text_a = example['text_a'] + tokenizer.sep_token + example['text_c']
        else:
            text_a = example['text_a']

        for j, choice in enumerate(example['text_b']):
            inp = tokenizer.encode_plus(text_a, choice, add_special_tokens=True, 
                max_length=max_len_clip, pad_to_max_length=True)
            inp['label'] = example['label']
            choice_feats.append(inp)

            if ex_index < 2:
                logger.info(f"==== Example {ex_index}, Choice {j} ====")
                logger.info("label: {}".format(inp['label']))
                logger.info("input: {}".format(tokenizer.decode(inp['input_ids'])))

        features.append(choice_feats)

    return features


def convert_data_to_features(
    dataset,
    tokenizer: PreTrainedTokenizer,
    task_name,
    max_len_clip,
):
    max_seq_len = 0
    features, all_lengths = [], []
    for (ex_index, example) in tqdm(enumerate(dataset), desc="convert examples to features"):
        if task_name == 'sst2': 
            text_a = example['sentence']
        elif task_name == 'imdb': 
            text_a = example['text']
        inp = tokenizer.encode_plus(text_a, add_special_tokens=True, 
            max_length=max_len_clip, pad_to_max_length=True)

        inp['label'] = example['label']
        features.append(inp)
            
        if ex_index < 2:
            logger.info(f"========== Example {ex_index} ==========")
            logger.info("label: {}".format(inp['label']))
            logger.info("input: {}".format(tokenizer.decode(inp['input_ids'])))

    return features


def get_data_loader(tokenizer, task_name, split, max_len, batch_size, data_dir=None):
    task_name = task_name.lower()
    processor_map = {'piqa': PIQAProcessor, 'hellaswag': HellaSwagProcessor, 'cosmosqa': COSMOSQAProcessor, 
        'imdb': IMDBProcessor, 'sst2': GLUEProcessor} 

    # load if cached
    path = f'TensorDataset_{task_name}_{split}_{max_len}.pt'
    if os.path.exists(path):
        dataset = torch.load(path)
        logger.info(f"Loaded the cached file from {path}!")
    else:
        processor = processor_map[task_name]()
        if split == 'train':
            dataset = processor.get_train_examples(data_dir)
        elif split == 'dev':
            dataset = processor.get_dev_examples(data_dir)
        else:
            dataset = processor.get_test_examples(data_dir)

        mc_set = set(['cosmosqa', 'hellaswag', 'piqa'])
        if task_name in mc_set:
            features = convert_mc_data_to_features(dataset, tokenizer, task_name, max_len)
            # convert features to tensor
            all_input_ids = torch.LongTensor([[choice['input_ids'] for choice in feature] for feature in features])
            all_attn_mask = torch.LongTensor([[choice['attention_mask'] for choice in feature] for feature in features])
            all_token_type = torch.LongTensor([[choice['token_type_ids'] for choice in feature] for feature in features])
            all_labels = torch.LongTensor([feature[0]['label'] for feature in features])
        else:
            features = convert_data_to_features(dataset, tokenizer, task_name, max_len)
            # convert features to tensor
            all_input_ids = torch.LongTensor([feature['input_ids'] for feature in features])
            all_attn_mask = torch.LongTensor([feature['attention_mask'] for feature in features])
            all_token_type = torch.LongTensor([feature['token_type_ids'] for feature in features])
            all_labels = torch.LongTensor([feature['label'] for feature in features])

        dataset = TensorDataset(all_input_ids, all_attn_mask, all_token_type, all_labels)
        torch.save(dataset, path)
        logger.info(f"Cached the TensorDataset to {path}!")

    # build dataloader
    if split == "train":
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader
