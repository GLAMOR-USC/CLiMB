import argparse
from collections import defaultdict
import datetime
import json
import logging
import os
import random
import sys
sys.path.insert(0, '.')
import time
import math
import shutil
import pickle as pkl
from PIL import Image
import copy
import pdb
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup
from transformers import BertTokenizer

from modeling import load_encoder_map
from configs.model_configs import model_configs
from configs.task_configs import task_configs
from utils.seed_utils import set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def train_vision(args, encoder, task_config, model_config, tokenizer, device):
    '''
    Train the encoder on the vision-only downstream tasks

    args: arguments provided by user
    encoder: trained encoder on upstream tasks
    task_config: dictionary containing task-specific configuration parameters for the current task
    model_config: dictionary containing model-specific configuration parameters
    tokenizer: tokenzier of the backbone encoder
    device: cuda/cpu
    '''

    # get upstream algo name for logging
    upstream_name = args.checkpoint_name.split('/')[-2]
    for short in ['adapter', 'ewc', 'replay', 'sequent', 'bottom9']:
        if short in args.checkpoint_name:
            upstream_name += f"_{short}"
            break
    logger.info(f"Upstream Task: {upstream_name}")

    # config
    task_name = task_config['task_name']
    num_labels = task_config['num_labels']
    data_dir = task_config['data_dir']
    # coco-obj-cls (multi-label) uses percentage for low-shot learning; other single-label tasks: N-shot per class
    n_shot = args.num_shot if args.task_name == 'coco-cls' else int(args.num_shot)
    subsample_seed = args.subsample_seed
    output_dir = args.output_dir

    # Create model
    batch2inputs_converter = model_config['batch2inputs_converter']
    encoder_dim = model_config['encoder_dim']
    visual_input_type = model_config['visual_input_type']
    classifier_class = model_config['classifier_class']
    model = classifier_class(encoder=encoder, 
                             encoder_dim=encoder_dim, 
                             num_labels=num_labels)

    model.to(device)

    # Create dataloaders for training, validation, and test sets
    if args.task_name == 'imagenet':
        from data.vision_datasets.imagenet_dataset import get_data_loader
    elif args.task_name == 'places365':
        from data.vision_datasets.places365_dataset import get_data_loader
    elif args.task_name == 'inat2019':
        from data.vision_datasets.inat2019_dataset import get_data_loader
    elif args.task_name == 'coco-cls':
        from data.vision_datasets.coco_cls_dataset import get_data_loader
    else:
        raise NotImplementedError("get_data_loader not impelmented for this task!")

    eval_fn = eval_coco if args.task_name == 'coco-cls' else eval_acc

    train_dataloader = get_data_loader(
        args,
        data_dir,
        'train', 
        n_shot,
        subsample_seed
    )
    val_dataloader = get_data_loader(
        args,
        data_dir,
        'val',
        n_shot
    ) 
    test_dataloader = get_data_loader(
        args,
        data_dir,
        'test'
    ) 

    # Training hyperparameters
    num_epochs = task_config['num_epochs']
    lr = task_config['lr']
    adam_epsilon = task_config['adam_epsilon']
    weight_decay = task_config['weight_decay']
    warmup_ratio = task_config['warmup_ratio']

    # Create optimizer
    if args.task_name == 'coco-cls':
        loss_criterion = nn.BCEWithLogitsLoss()
    else:
        loss_criterion = nn.CrossEntropyLoss()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/vilt_utils.py#L236
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon, betas=(0.9, 0.98))
    # Create Scheduler
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/vilt_utils.py#L263
    max_steps = len(train_dataloader) * num_epochs
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_steps * warmup_ratio),
        num_training_steps=max_steps,
        lr_end=0,
        power=1,
    )


    best_score = 0
    model.zero_grad()
    model.train()
    for epoch in range(1, num_epochs+1):
        for step, batch in enumerate(tqdm(train_dataloader, desc='Training epoch {}'.format(epoch))):
            labels = batch['labels'].to(device)
            inputs = batch2inputs_converter(batch)

            logits = model(**inputs)
            loss = loss_criterion(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 50 == 0:
                print('loss:', loss.item())

        # Eval on the val set and update the best model
        if epoch > 5 and epoch%2 == 0:
            eval_score = eval_fn(args, model, val_dataloader, device, batch2inputs_converter)
            logger.info("Evaluation after epoch {}: {:.2f}".format(epoch, eval_score))

            if eval_score > best_score:
                logger.info("New best evaluation score: {:.2f}".format(eval_score))
                best_score = eval_score
                best_epoch = epoch
                best_model = copy.deepcopy(model)

    # eval the best model on the test set & write the results
    test_score = eval_fn(args, best_model, test_dataloader, device, batch2inputs_converter)
    write_results(n_shot, subsample_seed, best_score, test_score, best_epoch, task_name, upstream_name, output_dir)


def write_results(n_shot, subsample_seed, best_score, test_score, best_epoch, task_name, upstream_name, output_dir):
    '''
    Write the results to the .json file in output_dir
    '''

    tree = lambda: defaultdict(tree)
    all_scores = tree()
    out_fn = os.path.join(output_dir, f'{task_name}_{upstream_name}_results.json')
    # load previous results
    if os.path.exists(out_fn):
        with open(out_fn, "r") as f:
            rdict = json.load(f)
        for k, v in rdict.items():
            all_scores[k] = v
    # update current results
    all_scores[f'nshot-{n_shot}'][f'seed-{subsample_seed}'] = (test_score, best_score, best_epoch)
    with open(out_fn, "w") as outfile:
        outfile.write(json.dumps(all_scores))



def eval_coco(args, model, eval_dataloader, device, batch2inputs_converter):
    '''
    Evaluation for the MS-COCO object classification task, using F1-micro as the metric

    model: the trained model
    eval_dataloader: dataloader for evaluation
    device: cuda/cpu
    batch2inputs_converter: a model-specific fuction that converts inputs to the format the model takes
    '''

    model.eval()
    act_fn = nn.Sigmoid()

    all_labels = torch.zeros((len(eval_dataloader.dataset), 80), dtype=torch.bool)
    all_preds = all_labels.clone()
    offset = 0
    for step, batch in enumerate(tqdm(eval_dataloader, desc='Evaluating...')):
        labels = batch['labels']
        inputs = batch2inputs_converter(batch)
        with torch.no_grad():
            logits = model(**inputs)
            preds = act_fn(logits) > 0.5

        all_labels[offset: offset+len(labels)] = labels.bool().cpu()
        all_preds[offset: offset+len(labels)] = preds.cpu()

        offset += len(labels)

    f1 = f1_score(all_labels, all_preds, average='micro')*100.0
    logger.info(f'Eval_F1: {f1:.3f}')

    model.train()
    return f1


def eval_acc(args, model, eval_dataloader, device, batch2inputs_converter):
    '''
    Evaluation on the dev set and test set, using accuracy as the metric

    model: the trained model
    eval_dataloader: dataloader for evaluation
    device: cuda/cpu
    batch2inputs_converter: a model-specific fuction that converts inputs to the format the model takes
    '''

    model.eval()
    eval_score = 0
    for step, batch in enumerate(tqdm(eval_dataloader, desc='Evaluating...')):
        labels = batch['labels']
        inputs = batch2inputs_converter(batch)
        with torch.no_grad():
            logits = model(**inputs)

        batch_scores = (logits.argmax(-1).cpu() == labels)
        eval_score += batch_scores.sum().item()

    eval_score = eval_score/len(eval_dataloader.dataset)*100.0
    logger.info(f'Eval_acc: {eval_score:.3f}')

    model.train()
    return eval_score



def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the vision-only task.")
    parser.add_argument("--encoder_name", default=None, type=str, required=True, choices=['vilt'],
                        help="The name of the base pretrained encoder.")
    parser.add_argument("--model_catog", default='vilt-vl', type=str,
                        help="The catogory for model class.")
    parser.add_argument("--checkpoint_name", default=None, type=str, required=True,
                        help="Name of the checkpoint model load.")
    parser.add_argument("--pretrained_model_name", default="dandelin/vilt-b32-mlm", type=str,
                        help="Name of the pretrained model")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Name of output directory, where all experiment results and checkpoints are saved.")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for dataloader")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    # only used by few-shot downstream tasks
    parser.add_argument("--num_shot", type=float,
                        help="Number of training data per class OR the ratio of the original training set")
    parser.add_argument("--subsample_seed", type=int,
                        help="Random seed for few-shot sampling.")

    
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    set_seed(args)


    # Load the Encoder model
    model_config = model_configs[args.model_catog]
    load_encoder_method = load_encoder_map[args.encoder_name]
    encoder = load_encoder_method(args.checkpoint_name, device, args.pretrained_model_name)

    results = []
    logger.info("-"*100)
    logger.info("Training models on downstream vision-only tasks...")

    # Load the correct training method for current CL task, and call the training method
    task_config = task_configs[args.task_name]
    logger.info("-"*100)
    train_vision(args, encoder, task_config, model_config, tokenizer, device)

if __name__ == '__main__':
    main()
