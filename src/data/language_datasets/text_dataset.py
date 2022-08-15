from tqdm import tqdm
import pickle
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


class LanguageDataset(Dataset):
    def __init__(self, processor, data_dir, split, task_name, n_shot=None, seed=None):
        """
        Initiate the Dataset for language-only tasks - loads all the sentences and the corresponding labels into self.data

        data_dir: path containing annotations and images
        split: either train/val/test
        task_name: the name of the language-only task
        n_shot: n-shot per class for classification tasks; number of examples for multiple-choice tasks
        seed: random seed for low-shot subsampling
        """

        self.task_name = task_name

        if split == 'train':
            data = processor.get_train_examples(data_dir) # type: list
            n_all = len(data)
            np.random.seed(seed)
            if task_name in ['commonsenseqa', 'hellaswag', 'piqa']: # multiple-choice
                self.sel_ids = set(np.random.choice(n_all, n_shot, replace=False))
            else: # sequence classification; balance classes
                labels = np.array([dt['label'] for dt in data])
                all_pos_ids = np.where(labels==1)[0]
                all_neg_ids = np.where(labels==0)[0]
                pos_sel_ids = set(np.random.choice(all_pos_ids, n_shot, replace=False))
                neg_sel_ids = set(np.random.choice(all_neg_ids, n_shot, replace=False))
                self.sel_ids = pos_sel_ids.union(neg_sel_ids)
                assert labels[np.array(list(self.sel_ids))].mean() == 0.5, "error in class balance"
            self.data = [dt for i, dt in enumerate(data) if i in self.sel_ids]

        elif split == 'val':
            self.data = processor.get_dev_examples(data_dir)
        else:
            self.data = processor.get_test_examples(data_dir)
        self.n_examples = len(self.data)
        logger.info(f"# Data in {split} set: {self.n_examples}")

    def __len__(self):
        return self.n_examples

    def __getitem__(self, index):
        example = self.data[index]
        if self.task_name == 'sst2': 
            return example['sentence'], example["label"]
        elif self.task_name == 'imdb': 
            return example['text'], example["label"]
        else:
            return example['text_a'], example['text_b'], example["label"]


def get_data_loader(tokenizer, task_name, split, max_len, batch_size, num_workers, data_dir=None, n_shot=None, seed=None):
    """
    Retrun a torch.utils.data.DataLoader for the dataset

    tokenizer: tokenizer of the backbone model
    task_name: the task name of the current language-only task
    split: either train/val/test split
    max_len: the maximum sequence length of the current task
    batch_size: how many samples per batch to load
    num_workers: how many subprocesses to use for data loading
    data_dir: path containing the dataset
    n_shot: n-shot per class for classification tasks; number of examples for multiple-choice tasks
    seed: random seed for low-shot subsampling
    """

    task_name = task_name.lower()
    processor_map = {'piqa': PIQAProcessor, 'hellaswag': HellaSwagProcessor, 'commonsenseqa': CommonsenseQAProcessor, 
        'imdb': IMDBProcessor, 'sst2': GLUEProcessor} 
    processor = processor_map[task_name]()

    dataset = LanguageDataset(processor, data_dir, split, task_name, n_shot, seed)

    # build dataloader
    dataloader = DataLoader(
        dataset, 
        shuffle=(split=='train'), 
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return dataloader
