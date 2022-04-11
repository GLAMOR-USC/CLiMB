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
from data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset
from modeling import load_encoder_map, continual_learner_map
from configs.model_configs import model_configs
from utils.seed_utils import set_seed


import copy
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
import pickle as pkl
import sklearn.metrics
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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


def plots(dataname, val_losses, train_losses, val_recall, train_recall, val_f1measure, train_f1measure, val_avgprecision, train_avgprecision, percent):

    plt.figure(figsize = (20, 4))
    plt.subplot(1, 4, 1)
    plt.plot(val_losses, 'bo-', label = 'val-loss')
    plt.plot(train_losses, 'ro-', label = 'train-loss')
    plt.grid('on')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation', 'training'], loc='upper left')

    plt.subplot(1, 4, 2)
    plt.plot(val_recall, 'bo-', label = 'val-recall')
    plt.plot(train_recall, 'ro-', label = 'train-recall')
    plt.ylabel('Recall')
    plt.grid('on')
    plt.xlabel('epoch')
    plt.legend(['validation', 'training'], loc='upper left')

    plt.subplot(1, 4, 3)
    plt.plot(val_f1measure, 'bo-', label = 'val-f1measure')
    plt.plot(train_f1measure, 'ro-', label = 'train-f1measure')
    plt.ylabel('F1measure')
    plt.grid('on')
    plt.xlabel('epoch')
    plt.legend(['validation', 'training'], loc='upper right')
    #plt.savefig( dataname + '_plots.png')
    #plt.show()

    plt.subplot(1, 4, 4)
    plt.plot(val_avgprecision, 'bo-', label = 'val-avgprecision')
    plt.plot(train_avgprecision, 'ro-', label = 'train-avgprecision')
    plt.ylabel('Avgprecision')
    plt.grid('on')
    plt.xlabel('epoch')
    plt.legend(['validation', 'training'], loc='upper right')
    plt.savefig( dataname + '_' + str(percent) + '_plots.png')
    plt.show()
    
def save_image_detection(model, mscoco_detection_val_dataloader, device ):
    #ids : 0, len()/4, 2*len()/4, 3*len()/4
    #model.eval()
    # I need the model + image
    print('hi')

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
          #temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_pred[i])
          temp+= np.nan_to_num(sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_true[i]))
      return np.nan_to_num(temp/ y_true.shape[0])


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
    #print(precision_den.shape)
    
    # precision averaged over all training examples
    #avg_precision = np.mean(precision_num/precision_den)
    avg_precision = np.mean(np.nan_to_num(precision_num/precision_den))
    
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

def train_mscoco_detection(args, model, task_configs, model_config, device, percent = 1):

    mscoco_detection_config   = task_configs['ms-coco_detection']
    mscoco_config = task_configs['ms-coco']
    images_dir      = os.path.join(args.mcl_data_dir, mscoco_config['data_dir'])
    annotation_dir  = os.path.join(args.mcl_data_dir, mscoco_detection_config['annotation_dir'])
    num_labels      = mscoco_detection_config['num_labels']
    visual_mode     = model_config['visual_mode']

    mscoco_images_dataset = MSCOCOImagesDataset(images_dir)

    mscoco_detection_train_dataloader = datasets.build_mscoco_detection_dataloader(args, 
                                                                                mscoco_images_dataset, 
                                                                                annotation_dir, 
                                                                                split='train', 
                                                                                visual_mode=visual_mode,
                                                                                percent = percent)

    mscoco_detection_val_dataloader = datasets.build_mscoco_detection_dataloader(args, 
                                                                                mscoco_images_dataset, 
                                                                                annotation_dir, 
                                                                                split='val', 
                                                                                visual_mode=visual_mode)
    # Create model
    model.to(device)

    # Training hyperparameters
    num_epochs      = mscoco_detection_config['num_epochs']
    lr              = mscoco_detection_config['lr']
    adam_epsilon    = mscoco_detection_config['adam_epsilon']
    weight_decay    = mscoco_detection_config['weight_decay']

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
    
    
    train_losses, train_recall, train_f1measure, train_avgprec = [], [], [], []
    val_losses, val_recall, val_f1measure, val_avgprec = [], [], [], []

    logger.info("Training model on MS-COCO Object Detection, with {}% training data".format(percent*100))
    for epoch in range(num_epochs):
        #val_losses_, val_recall_, val_f1measure_, eval_score  = eval_mscoco(args, model, mscoco_detection_val_dataloader, device)
        # Training loop for epoch
        ## Save the plots per epoch
        emr_total, recall_total, f1measure_total,avprecision_total, cumtrainloss, count = 0, 0, 0, 0, 0, 0
        #plots('../results/detection', val_losses, train_losses, val_recall, train_recall, val_f1measure, train_f1measure, val_avgprec, train_avgprec, percent)
        t = tqdm(mscoco_detection_train_dataloader,desc='Training epoch {}'.format(epoch+1))
        for step, batch in enumerate(t):
            
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
            output = model(task_key='ms-coco_detection', **inputs)
            logits = output[1]
            # https://github.com/dandelin/ViLT/blob/master/vilt/modules/objectives.py#L317
            loss = loss_criterion(logits, target) * target.shape[1]
            
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            cumtrainloss += loss.item()
            count +=1
            #if (step + 1) % 100 == 0:
            #    exit(0)
            ## Metrics for training
            y_true = target.cpu().detach().numpy()
            y_pred = torch.sigmoid(logits.cpu()).detach().numpy()
            #print(np.unique(y_true), np.unique(y_pred))
            #print(y_true.shape, y_pred.shape)
            y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)
            emr_total           += emr(y_true, y_pred)
            #recall_total        += sklearn.metrics.recall_score(y_true, y_pred, average='samples') #Recall(y_true, y_pred)
            recall_total        += Recall(y_true, y_pred) #
            #f1measure_total     += sklearn.metrics.f1_score(y_true, y_pred, average='samples') #F1Measure(y_true, y_pred)
            f1measure_total     += F1Measure(y_true, y_pred) #F1Measure(y_true, y_pred)
            #avprecision_total   += sklearn.metrics.precision_score(y_true, y_pred, average='samples') #example_based_precision(y_true, y_pred)
            avprecision_total   += example_based_precision(y_true, y_pred) #example_based_precision(y_true, y_pred)

            t.set_postfix(loss          = cumtrainloss/count, 
                            emr         = emr_total/count, 
                            recall      = recall_total/count, 
                            f1measure   = f1measure_total/count,
                            avgprecision = avprecision_total/count) 

        train_losses.append(cumtrainloss/count)
        train_recall.append(recall_total/count)
        train_f1measure.append(f1measure_total/count)
        train_avgprec.append(avprecision_total/count)
        ##Do evaluation step once training is working 
        val_losses_, val_recall_, eval_score, val_avgprecision_ = eval_mscoco_detection(args, model, mscoco_detection_val_dataloader, device)
        
        val_losses.append(val_losses_)
        val_recall.append(val_recall_)
        val_f1measure.append(eval_score) ## f1_measure
        val_avgprec.append(val_avgprecision_)

        ## Save the plots per epoch
        #plots('../results/detection', val_losses, train_losses, val_recall, train_recall, val_f1measure, train_f1measure, val_avgprec, train_avgprec, percent)

        logger.info("Evaluation after epoch {}: {:.2f}".format(epoch+1, eval_score))
        if eval_score > best_score:
            logger.info("New best evaluation score: {:.2f}".format(eval_score))
            best_score = eval_score
            best_model['epoch'] = epoch
            best_model['model'] = copy.deepcopy(model)
            best_model['loss']  = loss
            best_model['eval_score'] = eval_score
            best_model['percent'] = percent
            #print('eval_score: ', best_model['eval_score'], ', percent: ', percent)

    ## Now save the best model:
    path_model = '../results/'+ str(percent) +'image_classification.pt'
    torch.save({
            'epoch': best_model['epoch'],
            'model_state_dict': best_model['model'].state_dict(),
            'optimizer_state_dict': best_model['optimizer_state'], #optimizer.state_dict(),
            'loss': best_model['loss'],
            }, path_model)
    
    ## Save the plots per epoch
    #plots('../results/detection', val_losses, train_losses, val_recall, train_recall, val_f1measure, train_f1measure, val_avgprec, train_avgprec, percent)
    #print('eval_score: ', best_model['eval_score'], ', percent: ', percent)
    logger.info("Best validation F1 = {:.2f}, after epoch {}".format(best_model['eval_score'], best_model['epoch']))

    ##
    #file_mscoco = open('../results/results.txt', 'a')
    #file_mscoco.write('percent: ' + str(percent) + ', eval_score: ' + str(best_model['eval_score']) +  ', percent: ' + str(percent) + '\n')
    #file_mscoco.close()


def eval_mscoco_detection(args, model, mscoco_detection_val_dataloader, device):
    #loss
    loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    
    model.eval()
    eval_score = 0
    emr_total_, recall_total_, f1measure_total_, avprecision_total_, cumloss_, count_ = 0, 0, 0, 0, 0, 0
    t = tqdm(mscoco_detection_val_dataloader,desc='Val ')
    for step, batch in enumerate(t):
        images = batch['images']
        targets = batch['targets']
        #print(targets.shape)
        #print(images.shape)
        msg_text = ["" for a in range(0, len(images))]
        #msg_text  = torch.from_numpy(msg_text) 
        #print(images.shape, msg_text)
        ##torch.zeros((1, images.shape[0]), dtype=torch.long)
        
        count_ +=1
        inputs = {'images': images,'texts': msg_text} #batch2inputs_converter(batch)
        target = targets.to(device)

        #output = model(images=images, texts=texts)      # TODO: Create abstraction that can convert batch keys into model input keys for all models
        with torch.no_grad():
            output = model(task_key='ms-coco_detection', **inputs)
        logits = output[1]
        loss = loss_criterion(logits, target) * target.shape[1]
        cumloss_ += loss.item()

        y_true = target.cpu().detach().numpy()
        y_pred = torch.sigmoid(logits.cpu()).detach().numpy()
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        emr_total_           += emr(y_true, y_pred)
        #recall_total_        += sklearn.metrics.recall_score(y_true, y_pred, average='samples') #Recall(y_true, y_pred)
        recall_total_        += Recall(y_true, y_pred)
        #f1measure_total_     += sklearn.metrics.f1_score(y_true, y_pred, average='samples') #F1Measure(y_true, y_pred)
        f1measure_total_     += F1Measure(y_true, y_pred)
        #avprecision_total_   += sklearn.metrics.precision_score(y_true, y_pred, average='samples') #example_based_precision(y_true, y_pred)
        avprecision_total_   += example_based_precision(y_true, y_pred)
        t.set_postfix(loss          = cumloss_/count_, 
                        emr         = emr_total_/count_, 
                        recall      = recall_total_/count_, 
                        f1measure   = f1measure_total_/count_,
                        avgprecision= avprecision_total_/count_) 
    val_losses      = cumloss_/count_
    val_recall      = recall_total_/count_
    val_f1measure   = f1measure_total_/count_
    val_avprecision      = avprecision_total_/count_ #eval_score/len(vqa_val_dataloader.dataset)*100.0

    model.train()
    return val_losses, val_recall, val_f1measure, val_avprecision

def main():

    logger = logging.getLogger(__name__)

    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    from configs.task_configs import task_configs
    import copy


    class Args:
        def __init__(self):
            self.batch_size = 10
            self.shuffle = True
            self.num_workers = 2
            self.encoder_name = 'vilt'
            self.mcl_data_dir = '/data/datasets/MCL/'
            self.pretrained_model_name = 'dandelin/vilt-b32-mlm'
            self.seed = 42
            self.visual_mode = 'pil-image' #'raw'
    args = Args()

    set_seed(args)
    # Load the correct Encoder model, based on encoder_name argument
    model_config = model_configs[args.encoder_name]
    load_encoder_method = load_encoder_map[args.encoder_name]
    encoder = load_encoder_method(args.pretrained_model_name, device)
    continual_learner_class = continual_learner_map[args.encoder_name]
    model = continual_learner_class(['ms-coco_detection'], encoder, model_config['encoder_dim'], task_configs)

    train_mscoco_detection(args, model, task_configs, model_config, device, percent = 0.01)
    #train_mscoco_detection(args, model, task_configs, model_config, device, percent = 0.05)
    #train_mscoco_detection(args, model, task_configs, model_config, device, percent = 0.1)
    #train_mscoco_detection(args, model, task_configs, model_config, device, percent = 0.5)
    #train_mscoco_detection(args, model, task_configs, model_config, device, percent = 1)

if __name__ == '__main__':
    main()
