import os
import pdb 
import sys

#sys.path.insert(0, '.')
### dataset
import logging
import numpy as np

import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup

from data.vision_datasets import cocodetection_dataset as datasets 
### image_dataset -- tejas
from data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset
from configs.model_configs import model_configs
from utils.seed_utils import set_seed
import copy

import pdb
from tqdm import tqdm
import pickle as pkl

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)


logger = logging.getLogger(__name__)

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


def emr(y_true, y_pred):
    n = len(y_true)
    row_indicators = np.all(y_true == y_pred, axis = 1) # axis = 1 will check for equality along rows.
    exact_match_count = np.sum(row_indicators)
    return exact_match_count/n

def Recall(y_true, y_pred):
      temp = 0
      for i in range(y_true.shape[0]):
          if sum(y_pred[i]) == 0:
              continue
          temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_pred[i])
      return temp/ y_true.shape[0]


def F1Measure(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        temp+= (2*sum(np.logical_and(y_true[i], y_pred[i])))/ (sum(y_true[i])+sum(y_pred[i]))
    return temp/ y_true.shape[0]


def example_based_precision(y_true, y_pred):
    """
    precision = TP/ (TP + FP)
    """
    
    # Compute True Positive 
    precision_num = np.sum(np.logical_and(y_true, y_pred), axis = 1)
    
    # Total number of pred true labels
    precision_den = np.sum(y_pred, axis = 1)
    
    # precision averaged over all training examples
    avg_precision = np.mean(precision_num/precision_den)
    
    return avg_precision


'''
#print(F1Measure(y_true, y_pred))
def one_zero_loss(y_true, y_pred):
    n = len(y_true)
    row_indicators = np.logical_not(np.all(y_true == y_pred, axis = 1)) # axis = 1 will check for equality along rows.
    not_equal_count = np.sum(row_indicators)
    return not_equal_count/n


def hamming_loss(y_true, y_pred):
    """
	XOR TT for reference - 
	
	A  B   Output
	
	0  0    0
	0  1    1
	1  0    1 
	1  1    0
	"""
    hl_num = np.sum(np.logical_xor(y_true, y_pred))
    hl_den = np.prod(y_true.shape)
    
    return hl_num/hl_den


def example_based_accuracy(y_true, y_pred):
    
    # compute true positives using the logical AND operator
    numerator = np.sum(np.logical_and(y_true, y_pred), axis = 1)

    # compute true_positive + false negatives + false positive using the logical OR operator
    denominator = np.sum(np.logical_or(y_true, y_pred), axis = 1)
    instance_accuracy = numerator/denominator

    avg_accuracy = np.mean(instance_accuracy)
    return avg_accuracy

'''


def train_mscoco(args, encoder, task_configs, model_config, device):

    mscoco_config   = task_configs['ms-coco']
    images_dir      = mscoco_config['data_dir']
    annotation_dir  = mscoco_config['annotation_dir']
    num_labels      = mscoco_config['num_labels']
    visual_mode     = model_config['visual_mode']

    mscoco_images_dataset = MSCOCOImagesDataset(images_dir)

    mscoco_detection_train_dataloader = datasets.build_mscoco_detection_dataloader(args, 
                                                                                    images_dir, 
                                                                                    annotation_dir, 
                                                                                    split='train', 
                                                                                    visual_mode=visual_mode)

    mscoco_detection_val_dataloader = datasets.build_mscoco_detection_dataloader(args, 
                                                                                    images_dir, 
                                                                                    annotation_dir, 
                                                                                    split='val', 
                                                                                    visual_mode=visual_mode)
    # Create model
    encoder_dim         = model_config['encoder_dim']
    visual_mode         = model_config['visual_mode']
    classifier_class    = model_config['classifier_class']
    model = classifier_class(encoder=encoder, 
                             encoder_dim=encoder_dim, 
                             num_labels=num_labels)
    #batch2inputs_converter = model_config['batch2inputs_converter']
    model.to(device)


    # Training hyperparameters
    num_epochs      = mscoco_config['num_epochs']
    lr              = mscoco_config['lr']
    adam_epsilon    = mscoco_config['adam_epsilon']
    weight_decay    = mscoco_config['weight_decay']

    # Create optimizer
    loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/vilt_utils.py#L236
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon, betas=(0.9, 0.98))
    # Create Scheduler
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/vilt_utils.py#L263
    max_steps = len(mscoco_detection_train_dataloader) * num_epochs
    warmup_ratio = 0.1 # TODO remove hard code
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_steps * warmup_ratio),
        num_training_steps=max_steps,
        lr_end=0,
        power=1,
    )

    best_score = 0
    best_model = {
        'epoch': 0,
        'loss':  0,
        'model': copy.deepcopy(model), #model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }

    model.zero_grad()
    model.train()
    #num_epochs = 1
    emr_total, recall_total, f1measure_total,avprecision_total, cumloss, count = 0, 0, 0, 0, 0, 0
    
    print("I am ready to start training by epochs")
    for epoch in range(num_epochs):
        # Training loop for epoch
        t = tqdm(mscoco_detection_train_dataloader,desc='Training epoch {}'.format(epoch))
        for step, batch in enumerate(mscoco_detection_train_dataloader):
            
            images = batch['images']
            targets = batch['targets']
            #print(targets.shape)
            #print(images.shape)
            msg_text = ["" for a in range(0, len(images))]
            #msg_text  = torch.from_numpy(msg_text) 
            #print(images.shape, msg_text)
            ##torch.zeros((1, images.shape[0]), dtype=torch.long)
            
           
            inputs = {'images': images,'texts': msg_text} #batch2inputs_converter(batch)
            target = targets.to(device)

            #output = model(images=images, texts=texts)      # TODO: Create abstraction that can convert batch keys into model input keys for all models
            output = model(**inputs)
            logits = output[1]
            # https://github.com/dandelin/ViLT/blob/master/vilt/modules/objectives.py#L317
            loss = loss_criterion(logits, target) * target.shape[1]
            
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            cumloss += loss.item()
            count +=1
            #if (step + 1) % 100 == 0:
            #    exit(0)
            ## Metrics for training
            y_true = target.cpu().detach().numpy()
            y_pred = torch.sigmoid(logits.cpu()).detach().numpy()
            #print(np.unique(y_true), np.unique(y_pred))
            #print(y_true.shape, y_pred.shape)
            emr_total           += emr(y_true, y_pred)
            recall_total        += Recall(y_true, y_pred)
            f1measure_total     += F1Measure(y_true, y_pred)
            avprecision_total   += example_based_precision(y_true, y_pred)

            t.set_postfix(loss          = cumloss/count, 
                            emr         = emr_total/count, 
                            recall      = recall_total/count, 
                            f1measure   = f1measure_total/count,
                            avgprecision = avprecision_total/count) 
        ##Do evaluation step once training is working 
        eval_score = eval_mscoco(args, model, mscoco_detection_val_dataloader, device)
        logger.info("Evaluation after epoch {}: {:.2f}".format(epoch+1, eval_score))
        if eval_score > best_score:
            logger.info("New best evaluation score: {:.2f}".format(eval_score))
            best_score = eval_score
            best_model['epoch'] = epoch
            best_model['model'] = copy.deepcopy(model)
            best_model['loss']  = loss
        
    ## Now save the best model:
    path_model = '../../results/image_classification.pt'
    torch.save({
            'epoch': best_model['epoch'],
            'model_state_dict': best_model['model'].state_dict(),
            'optimizer_state_dict': best_model['optimizer_state'], #optimizer.state_dict(),
            'loss': best_model['loss'],
            }, path_model)

def eval_mscoco(args, model, mscoco_detection_val_dataloader, device):

    model.eval()
    eval_score = 0
    emr_total, recall_total, f1measure_total,avprecision_total, cumloss, count = 0, 0, 0, 0, 0, 0
    t = tqdm(mscoco_detection_val_dataloader,desc='Val epoch {}'.format(epoch))
    for step, batch in enumerate(mscoco_detection_val_dataloader):
        images = batch['images']
        targets = batch['targets']
        #print(targets.shape)
        #print(images.shape)
        msg_text = ["" for a in range(0, len(images))]
        #msg_text  = torch.from_numpy(msg_text) 
        #print(images.shape, msg_text)
        ##torch.zeros((1, images.shape[0]), dtype=torch.long)
        
        
        inputs = {'images': images,'texts': msg_text} #batch2inputs_converter(batch)
        target = targets.to(device)

        #output = model(images=images, texts=texts)      # TODO: Create abstraction that can convert batch keys into model input keys for all models
        with torch.no_grad():
            output = model(**inputs)
        logits = output[1]
        
        y_true = target.cpu().detach().numpy()
        y_pred = torch.sigmoid(logits.cpu()).detach().numpy()
        
        emr_total           += emr(y_true, y_pred)
        recall_total        += Recall(y_true, y_pred)
        f1measure_total     += F1Measure(y_true, y_pred)
        avprecision_total   += example_based_precision(y_true, y_pred)
        t.set_postfix(loss          = cumloss/count, 
                        emr         = emr_total/count, 
                        recall      = recall_total/count, 
                        f1measure   = f1measure_total/count,
                        avgprecision= avprecision_total/count) 
    eval_score = avprecision_total/count #eval_score/len(vqa_val_dataloader.dataset)*100.0

    model.train()
    return eval_score

'''
import sklearn.metrics

print('Exact Match Ratio: {0}'.format(sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))
#Exact Match Ratio: 0.25

print('Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(y_true, y_pred))) 
#Hamming loss: 0.4166666666666667

#"samples" applies only to multilabel problems. It does not calculate a per-class measure, instead calculating the metric over the true and predicted classes 
#for each sample in the evaluation data, and returning their (sample_weight-weighted) average.

print('Recall: {0}'.format(sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples'))) 
#Recall: 0.375

print('Precision: {0}'.format(sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples')))
#Precision: 0.5

print('F1 Measure: {0}'.format(sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples'))) 
#F1 Measure: 0.41666666666666663
'''