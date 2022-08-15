import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderWrapper(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, **kwargs):
        pass


class ContinualLearner(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, **kwargs):
        pass

    def get_encoder(self):
        pass