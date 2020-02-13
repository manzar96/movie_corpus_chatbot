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
        x = torch.sqrt(x)
        return x


class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[-1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[-1], self._pool_size)
        m, i = x.view(*x.shape[:-1], x.shape[-1] // self._pool_size, self._pool_size).max(-1)
        return m

def max_out(x):
    # make sure s2 is even and that the input is 2 dimension
    if len(x.size()) == 2:
        s1, s2 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2 // 2, 2)
        x, _ = torch.max(x, 2)

    elif len(x.size()) == 3:
        s1, s2, s3 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2, s3 // 2, 2)
        x, _ = torch.max(x, 3)

    return x


class Maxout2(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m



