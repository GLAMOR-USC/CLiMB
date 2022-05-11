import random
import wandb
import logging

from torch import nn
from torch.optim import AdamW

logger = logging.getLogger(__name__)

def vqa_replay_step(model, replay_memory, task_configs, batch2inputs_converter, device):

    vqa_config = task_configs['vqa']
    # Training hyperparameters
    num_epochs = vqa_config['num_epochs']
    lr = vqa_config['lr']
    adam_epsilon = vqa_config['adam_epsilon']
    weight_decay = vqa_config['weight_decay']

    # Create optimizer
    loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/vilt_utils.py#L236
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon, betas=(0.9, 0.98))
    
    replay_batch = replay_memory.sample_replay_batch('vqa')
    inputs = batch2inputs_converter(replay_batch)
    target = replay_batch['target_scores'].to(device)

    #output = model(images=images, texts=texts)      # TODO: Create abstraction that can convert batch keys into model input keys for all models
    output = model(task_key='vqa', **inputs)
    logits = output[1]
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/objectives.py#L317
    loss = loss_criterion(logits, target) * target.shape[1]

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    wandb.log({'vqa': {'loss': loss.item()}})

    return loss.item()

def nlvr2_replay_step(model, replay_memory, task_configs, batch2inputs_converter, device):

    nlvr_config = task_configs['nlvr2']
    # Training hyperparameters
    num_epochs = nlvr_config['num_epochs']
    lr = nlvr_config['lr']
    adam_epsilon = nlvr_config['adam_epsilon']
    weight_decay = nlvr_config['weight_decay']
    warmup_ratio = nlvr_config['warmup_ratio']

    # Create optimizer
    loss_criterion = nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/vilt_utils.py#L236
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon, betas=(0.9, 0.98))
    
    replay_batch = replay_memory.sample_replay_batch('nlvr2')
    target = replay_batch['labels'].to(device)
    inputs = batch2inputs_converter(replay_batch)

    output = model(task_key='nlvr2', **inputs)
    logits = output[1]
    loss = loss_criterion(logits, target)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    wandb.log({'nlvr': {'loss': loss.item()}})

    return loss.item()

def snli_ve_replay_step(model, replay_memory, task_configs, batch2inputs_converter, device):

    snli_ve_config = task_configs['snli-ve']
    # Training hyperparameters
    num_epochs = snli_ve_config['num_epochs']
    lr = snli_ve_config['lr']
    adam_epsilon = snli_ve_config['adam_epsilon']
    weight_decay = snli_ve_config['weight_decay']

    # Create optimizer
    loss_criterion = nn.CrossEntropyLoss(reduction='mean')
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/vilt_utils.py#L236
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon, betas=(0.9, 0.98))
    
    replay_batch = replay_memory.sample_replay_batch('snli-ve')
    inputs = batch2inputs_converter(replay_batch)
    labels = replay_batch['labels'].to(device)

    #output = model(images=images, texts=texts)      # TODO: Create abstraction that can convert batch keys into model input keys for all models
    output = model(task_key='snli-ve', **inputs)
    logits = output[1]
    # https://github.com/dandelin/ViLT/blob/master/vilt/modules/objectives.py#L317
    loss = loss_criterion(logits, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    wandb.log({'snli-ve': {'loss': loss.item()}})

    return loss.item()


REPLAY_STEP_METHOD_MAP = {'vqa': vqa_replay_step,
                          'nlvr2': nlvr2_replay_step,
                          'snli-ve': snli_ve_replay_step   
                        }

class ExperienceReplayMemory:

    def __init__(self):
        self.memory_buffers = {}
        self.replay_step_method = {}

    def add_task_memory_buffer(self, args, task_key, task_config, train_dataset, memory_percentage, sampling_strategy):

        task_buffer = TaskMemoryBuffer(args, task_key, task_config, train_dataset, memory_percentage, sampling_strategy)
        self.memory_buffers[task_key] = task_buffer
        self.replay_step_method[task_key] = REPLAY_STEP_METHOD_MAP[task_key]

    def do_replay(self):
        '''
        Return true if there are any tasks in the memory to do replay on, else False
        '''
        return True if len(self.memory_buffers) > 0 else False

    def sample_replay_task(self):
        previous_tasks = list(self.memory_buffers.keys())
        sampled_previous_task = random.choice(previous_tasks)
        return sampled_previous_task

    def sample_replay_batch(self, task_key):
        return self.memory_buffers[task_key].sample_replay_batch()

    def run_replay_step(self, task_key, **replay_args):
        replay_step_method = self.replay_step_method[task_key]
        return replay_step_method(replay_memory=self, **replay_args)

class TaskMemoryBuffer:

    '''
    Buffer of training examples that can be used for replay steps
    '''
    def __init__(self, args, task_key, task_config, train_dataset, memory_percentage, sampling_strategy):

        self.task_key = task_key
        self.task_name = task_config['task_name']
        self.batch_collate_fn = task_config['batch_collate_fn']

        self.dataset = train_dataset
        if task_key == 'nlvr2':
            self.batch_size = int(args.batch_size/2)
        else:
            self.batch_size = args.batch_size
        self.visual_mode = args.visual_mode

        self.memory_percentage = memory_percentage                      # Percent of training samples to store in memory
        assert self.memory_percentage < 1.0
        self.memory_size = int(memory_percentage*len(self.dataset))     # Number of training samples that are stored in memory
        self.sampling_strategy = sampling_strategy
        assert sampling_strategy in ['random']                      # Only random sampling for memory buffer implemented so far

        if self.sampling_strategy == 'random':
            train_idxs = list(range(len(self.dataset)))
            self.memory_idxs = random.sample(train_idxs, self.memory_size)

        elif self.sampling_strategy == 'random-balanced':
            raise NotImplementedError("Label-balanced sampling of replay memory is not yet implemented!")

        logger.info("Created {} replay memory buffer, with {} samples in the memory".format(self.task_name, len(self.memory_idxs)))

    def __len__(self):
        return len(self.memory_idxs)

    def sample_replay_batch(self):

        sampled_instances = random.sample(self.memory_idxs, self.batch_size)
        batch = self.batch_collate_fn([self.dataset[i] for i in sampled_instances], self.visual_mode)
        return batch
