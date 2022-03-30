from tqdm import tqdm
import pdb
import logging
import os
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, TensorDataset, Dataset

from data.language_datasets.text_processors import *

transformers.logging.set_verbosity_error()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def convert_mc_data_to_features(
    dataset,
    tokenizer,
    task_name,
    max_len_clip,
):
    max_seq_len = 0
    features, all_lengths = [], []
    for (ex_index, example) in tqdm(enumerate(dataset), desc="convert examples to features"):
        choice_feats = []
        for j, choice in enumerate(example['text_b']):
            if task_name == 'cosmosqa':
                text_b = example['text_c'] + tokenizer.sep_token + choice
            else:
                text_b = choice
            inp = tokenizer.encode_plus(example['text_a'], text_b, add_special_tokens=True, 
                max_length=max_len_clip, padding='max_length', truncation='longest_first')
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
    tokenizer,
    task_name,
    max_len_clip,
):
    max_seq_len = 0
    features, all_lengths = [], []
    for (ex_index, example) in tqdm(enumerate(dataset), desc="convert examples to features"):
        if task_name == 'sst2': 
            text_a = example['sentence']
            text_b = None
        elif task_name == 'imdb': 
            text_a = example['text']
            text_b = None
        inp = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, 
                max_length=max_len_clip, padding='max_length', truncation='longest_first')

        inp['label'] = example['label']
        features.append(inp)
            
        if ex_index < 2:
            logger.info(f"========== Example {ex_index} ==========")
            logger.info("label: {}".format(inp['label']))
            logger.info("input: {}".format(tokenizer.decode(inp['input_ids'])))

    return features


class LanguageDataset(Dataset):
    def __init__(self, processor, data_dir, split, task_name, is_fewshot=True):
        self.task_name = task_name
        mc_set = set(['cosmosqa', 'hellaswag', 'piqa'])
        self.is_mc = task_name in mc_set

        if split == 'train':
            self.data = processor.get_train_examples(data_dir) # type: list
            if is_fewshot: # subsample
                n_all = len(self.data)
                np.random.seed(2022)
                sel_ids = set(np.random.choice(n_all, int(n_all*0.1), replace=False))
                self.data = [dt for i, dt in enumerate(self.data) if i in sel_ids]
        elif split == 'val':
            self.data = processor.get_dev_examples(data_dir)
        else:
            self.data = processor.get_test_examples(data_dir)
        self.n_examples = len(self.data)
        print('# Data:', self.n_examples)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, index):
        example = self.data[index]
        if self.task_name == 'sst2': 
            text = example['sentence']
        elif self.task_name == 'imdb': 
            text = example['text']
        else: #TODO
            pdb.set_trace()

        return text, example["label"]


def get_data_loader(tokenizer, task_name, split, max_len, batch_size, cache_dir, data_dir=None):
    task_name = task_name.lower()
    processor_map = {'piqa': PIQAProcessor, 'hellaswag': HellaSwagProcessor, 'cosmosqa': COSMOSQAProcessor, 
        'imdb': IMDBProcessor, 'sst2': GLUEProcessor} 
    processor = processor_map[task_name]()

    dataset = LanguageDataset(processor, data_dir, split, task_name)

    # build dataloader
    dataloader = DataLoader(
        dataset, 
        shuffle=(split=='train'), 
        batch_size=batch_size,
        num_workers=4,
    )

    return dataloader
