import json
import pickle as pkl
import os

import torch

from collections import defaultdict, Counter
from utils.word_utils import normalize_word

def get_score(occurences):
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0

def create_vqa_labels(vqa_dir):

    train_annotations = json.load(open(os.path.join(vqa_dir, 'v2_mscoco_train2014_annotations.json')))['annotations']
    val_annotations = json.load(open(os.path.join(vqa_dir, 'v2_mscoco_val2014_annotations.json')))['annotations']

    all_major_answers = []
    for anno in train_annotations:
        all_major_answers.append(normalize_word(anno['multiple_choice_answer']))
    for anno in val_annotations:
        all_major_answers.append(normalize_word(anno['multiple_choice_answer']))
    counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}

    ans2label = {k: i for i, k in enumerate(counter.keys())}
    print("Number of labels: {}".format(len(ans2label)))

    pkl.dump(ans2label, open(os.path.join(vqa_dir, 'ans2label.pkl'), 'wb'))


def target_tensor(len, labels, scores):
    """ create the target by labels and scores """
    target = [0]*len
    for id, l in enumerate(labels):
        target[l] = scores[id]

    return torch.tensor(target)
''' # this seems more straightforward to me
def target_tensor(num_labels, labels, scores):
    """ create the target by labels and scores """
    target = torch.zeros(num_labels)
    target[labels] = torch.tensor(scores)

    return target
'''

if __name__ == '__main__':
    create_vqa_labels('/data/datasets/MCL/vqav2/')
