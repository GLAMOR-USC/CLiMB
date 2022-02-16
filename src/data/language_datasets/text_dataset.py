from tqdm import tqdm
import pdb
import csv
import glob
import json
import logging
import os
from typing import List
import pickle
import numpy as np



logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""
    def __init__(self):
        label_list = self._set_label_list()
        self.label_map = {label: i for i, label in enumerate(label_list)}

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    @classmethod
    def _to_example(self, example_id, text_a=None, text_b=None, text_c=None, label=None, desc=None):
        return {
            "example_id": example_id,
            "text_a": text_a,
            "text_b": text_b,
            "text_c": text_c,
            "label": label,
            "description": desc
        }

    @classmethod
    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    @classmethod
    def _read_jsonl(cls, input_file):
        data = []
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = json.loads(line)
                data.append(line)
        return data


class HellaSwagProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        path = os.path.join(data_dir, "hellaswag_train.jsonl")
        logger.info("Loading training set from {}".format(path))
        return self._create_examples(self._read_jsonl(path))

    def get_dev_examples(self, data_dir):
        path = os.path.join(data_dir, "hellaswag_val.jsonl")
        logger.info("Loading Dev set from {}".format(path))
        return self._create_examples(self._read_jsonl(path))


    def _set_label_list(self):
        return [0, 1, 2, 3]

    def _create_examples(self, data, has_label=True):
        examples = []
        for idx, dt in enumerate(data):
            examples.append(
                self._to_example(
                    example_id=idx,
                    text_a=[dt["ctx"]]*4,
                    text_b=dt["endings"],
                    label=self.label_map[dt["label"]] if has_label else None,
                    desc="Multiple-Choice; text_a: Ctx; text_b: ending"
                )
            )
        return examples


class PIQAProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        path = os.path.join(data_dir, "train.jsonl")
        label_path = os.path.join(data_dir, "train-labels.lst")
        logger.info("Loading training set from {}".format(path))
        return self._create_examples(self._read_jsonl(path), label_path)

    def get_dev_examples(self, data_dir):
        path = os.path.join(data_dir, "valid.jsonl")
        label_path = os.path.join(data_dir, "valid-labels.lst")
        logger.info("Loading Dev set from {}".format(path))
        return self._create_examples(self._read_jsonl(path), label_path)

    def _set_label_list(self):
        return ["0", "1"]

    def _create_examples(self, data, label_path, has_label=True):
        if has_label:
            labels = open(label_path, encoding="utf-8").read().splitlines()
        else:
            labels = np.zeros(len(data)) #dummy, not used

        examples = [
            self._to_example(
                example_id=idx,
                text_a=[dt["goal"]]*2,
                text_b=[dt["sol1"], dt["sol2"]],
                label=self.label_map[lb] if has_label else None,
                desc="Multiple-Choice; text_a: Ctx; text_b: Ans"
            )
            for idx, (dt, lb) in enumerate(zip(data, labels))
        ]
        return examples


class COSMOSQAProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        path = os.path.join(data_dir, "train.csv")
        logger.info("Loading Training set from {}".format(path))
        return self._create_examples(self._read_csv(path))

    def get_dev_examples(self, data_dir):
        path = os.path.join(data_dir, "valid.csv")
        logger.info("Loading Dev set from {}".format(path))
        return self._create_examples(self._read_csv(path))

    def _set_label_list(self):
        return ['0', '1', '2', '3']

    def _create_examples(self, data, has_label=True):
        examples = [
            self._to_example(
                example_id=line[0],
                text_a=[line[1]]*4,
                text_b=[line[2]]*4,
                text_c=[line[3], line[4], line[5], line[6]],
                label=self.label_map[line[7]] if has_label else None,
                desc="Multiple-Choice; text_a: Ctx; text_b: Ques; text_c: Ans"
            )
            for line in data[1:]  # we skip the line with the column names
        ]

        return examples


class IMDBProcessor():
    def __init__(self, cache_dir):
        from datasets import load_dataset
        dataset = load_dataset("imdb", cache_dir=cache_dir)

        # randomly sample dev_set_ids from annotated data
        n_labeled_data = 25000
        self.dev_ratio = 0.3
        np.random.seed(2022)
        self.dev_ids = set(np.random.choice(n_labeled_data, int(n_labeled_data*self.dev_ratio), replace=False))

        # train/val split
        self.train_data, self.dev_data = [], []
        for i, dt in enumerate(dataset['train']):
            if i in self.dev_ids:
                self.dev_data.append(dt)
            else:
                self.train_data.append(dt)

        self.test_data = dataset['test']

    def get_train_examples(self):
        return self.train_data

    def get_dev_examples(self):
        return self.dev_data


if __name__ == '__main__':
    from pprint import pprint
    logger.info("========= HellaSwag Examples =========") 
    processor = HellaSwagProcessor()
    dataset = processor.get_dev_examples(data_dir='hellaswag')
    pprint(dataset[0])
    pdb.set_trace()

    logger.info("========= PIQA Examples =========") 
    processor = PIQAProcessor()
    dataset = processor.get_dev_examples(data_dir='piqa')
    pprint(dataset[0])
    pdb.set_trace()

    logger.info("========= COSMOS QA Examples =========") 
    processor = COSMOSQAProcessor()
    dataset = processor.get_dev_examples(data_dir='cosmosqa')
    pprint(dataset[0])
    pdb.set_trace()

    logger.info("========= IMDB Examples =========") 
    processor = IMDBProcessor('cache_imdb')
    dataset = processor.get_dev_examples()
    pprint(dataset[0])
    pdb.set_trace()

