import torch
import torch.nn as nn
import numpy as np


class L2PoolingLayer(nn.Module):
    """
    Its an L2 pooling layer implemented for sequence L2 pooling on hidden
    state. It is described by Serban et al.2016 'Building end to end dialogue
    systems using ....'
    """
    def __init__(self):

        super(L2PoolingLayer, self).__init__()


    def forward(self, x):
        """
        input must be in form: [batch size , sequence length, features]
        """
        seq_len = x.shape[1]
        import ipdb;ipdb.set_trace()
        x = torch.pow(x, 2.)
        x = 1/seq_len * torch.sum(x, dim=1)
