import os
import wandb

class WandBLogger:

    def __init__(self):

        self.is_initialized = False

    def initialize(self, wandb_config, experiment_name):

        os.environ['WANDB_API_KEY'] = wandb_config['api_key']
        wandb.init(entity=wandb_config['entity'],
                   project=wandb_config['project_name'],
                   name=experiment_name)
        self.is_initialized = True
        self.log_freq = wandb_config['log_freq']

    def log(self, log_dict):

        if self.is_initialized:
            wandb.log(log_dict)

    def get_log_freq(self):
        if self.is_initialized:
            return self.log_freq
        else:
            return 100


wandb_logger = WandBLogger()