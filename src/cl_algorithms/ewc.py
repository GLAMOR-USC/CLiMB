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

    def add_task_parameters(self, task_key, model, task_trainer):

        task_config = task_configs[task_key]
        self.fisher_dict[task_key] = defaultdict(float)

        # Save model params
        self.param_dict[task_key] = {}
        for name, param in model.get_encoder().named_parameters():
            self.param_dict[task_key][name] = param.data.cpu().clone()

        assert task_key not in self.task_keys
        self.task_keys.append(task_key)

        optimizer = task_trainer.create_optimizer(model)
        batch2inputs_converter = task_trainer.batch2inputs_converter
        dataloader = task_trainer.get_train_dataloader()
        loss_criterion = task_trainer.loss_criterion

        optimizer.zero_grad()
        num_samples_completed = 0
        # Create fisher matrix
        for step, batch in enumerate(tqdm(dataloader, desc='Computing Fisher information matrix for {} checkpoint'.format(task_config['task_name']))):
            loss, output = task_trainer.train_step(model, batch)

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