import random

import torch

from configs.task_configs import task_configs

class EWC:

    def __init__(self, args):
        self.fisher_sample_size = args.ewc_fisher_sample_size
        self.ewc_loss_weight = args.ewc_loss_weight
        self.fisher_dict = {}
        self.param_dict = {}
        self.task_keys = []

    def add_task_parameters(self, task_key, model, train_args):

        self.fisher_dict[task_key] = defaultdict(float)
        # Save model params
        self.param_dict[task_key] = {}
        for name, param in model.get_encoder().named_parameters():
            self.param_dict[task_key][name] = param.data.cpu().clone()

        assert task_key not in self.task_keys
        self.task_keys.append(task_key)

        optimizer = train_args['optimizer']
        batch2inputs_converter = train_args['batch2inputs_converter']
        dataloader = train_args['dataloader']
        loss_criterion = train_args['loss_criterion']

        optimizer.zero_grad()
        num_samples_completed = 0
        # Create fisher matrix
        for step, batch in dataloader:
            inputs = batch2inputs_converter(batch)
            labels = batch['labels'].to(device)
            output = model(task_key=task_key, **inputs)
            logits = output[1]
            loss = loss_criterion(logits, labels)
            loss.backward()

            for name, param in model.get_encoder().named_parameters():
                self.fisher_dict[task_key][name] += param.grad.data.pow(2).cpu().clone()

            num_samples_completed += labels.shape[0]
            if num_samples_completed >= self.fisher_sample_size:
                break

        for name, param in model.get_encoder().named_parameters():
            self.fisher_dict[task_key][name] /=  num_samples_completed

    def compute_ewc_loss(self, model):

        ewc_task_key = random.sample(self.task_keys)
        ewc_loss = 0
        for name, param in model.get_encoder().named_parameters():
            ewc_param = self.param_dict[ewc_task_key][name].to(device)
            ewc_fisher = self.fisher_dict[ewc_task_key][name].to(device)
            ewc_loss += ewc_fisher*((param - ewc_param).pow(2).sum())
        return ewc_loss_weight*ewc_loss