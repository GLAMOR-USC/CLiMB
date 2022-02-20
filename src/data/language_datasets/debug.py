from pprint import pprint
from tqdm import tqdm
import pdb
import logging
import os
import pickle
import numpy as np
from transformers import BertTokenizer

from text_processors import *
from text_dataset import *

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def check_datasets(tokenizer):
    Processors = (IMDBProcessor, GLUEProcessor)
    names = ['imdb', 'sst2']
    for Proc, name in zip(Processors, names):
        logger.info(f"======================= {name} ==============================")
        processor = Proc()
        dataset = processor.get_test_examples()
        print('# Test', len(dataset))
        pprint(dataset[0])

        dataset = processor.get_train_examples()
        print('# Train:', len(dataset))
        pprint(dataset[0])

        dataset = processor.get_dev_examples()
        print('# Dev:', len(dataset))
        pprint(dataset[0])

        features = convert_data_to_features(dataset, tokenizer, name, 50)

    return features


def check_mc_datasets(tokenizer):
    Processors = (HellaSwagProcessor, PIQAProcessor, COSMOSQAProcessor, IMDBProcessor, GLUEProcessor)
    data_dirs = ['hellaswag', 'piqa', 'cosmosqa']
    for Proc, ddir in zip(Processors, data_dirs):
        logger.info(f"======================= {ddir} ==============================")
        processor = Proc()
        dataset = processor.get_test_examples(data_dir=ddir)
        print('# Test', len(dataset))
        pprint(dataset[0])

        dataset = processor.get_train_examples(data_dir=ddir)
        print('# Train:', len(dataset))
        pprint(dataset[0])

        dataset = processor.get_dev_examples(data_dir=ddir)
        print('# Dev:', len(dataset))
        pprint(dataset[0])

        features = convert_mc_data_to_features(dataset, tokenizer, ddir, 100)

    return features


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    features = check_datasets(tokenizer)
    features = check_mc_datasets(tokenizer)
    print(features[0])
    pdb.set_trace()
