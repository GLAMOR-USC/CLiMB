import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskTrainer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def train(self, **kwargs):
        pass

    def eval(self, **kwargs):
        pass
