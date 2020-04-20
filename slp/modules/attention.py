import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from slp.modules.feedforward import FF


def calc_scores(dk):
    def fn(q, k):
        return torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dk)
    return fn


class Attention(nn.Module):
    """Some Information about Attention"""
    def __init__(self,
                 attention_size=512,
                 input_size=None,
                 dropout=.1,
                 grad_checkpoint=False):
        super(Attention, self).__init__()
        if input_size is None:
            input_size = attention_size
        self.dk = input_size
        self.grad_checkpoint = grad_checkpoint
        self.k = nn.Linear(input_size, attention_size, bias=False)
        self.q = nn.Linear(input_size, attention_size, bias=False)
        self.v = nn.Linear(input_size, attention_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self._reset_parameters()

    def forward(self, x, queries=None, values=None, attention_mask=None):
        '''
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        '''
        if queries is None:
            queries = x
        if values is None:
            values = x
        k = self.k(x)  # (B, L, A)
        q = self.q(queries)  # (B, L, A)
        v = self.v(values)  # (B, L, A)

        # weights => (B, L, L)
        if self.grad_checkpoint:
            scores = checkpoint(calc_scores(self.dk), q, k)
        else:
            scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.dk)
        if attention_mask is not None:
            scores = scores + ((1 - attention_mask.unsqueeze(1)) * -1e5)
        scores = F.softmax(scores, dim=-1)
        scores = self.drop(scores)

        # out => (B, L, A)
        out = torch.bmm(scores, v)
        return out, scores

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.v.weight)


class MultiheadAttentionSerial(nn.Module):
    """Serial MultiheadAttention"""
    def __init__(self,
                 attention_size=512,
                 num_heads=8,
                 input_size=None,
                 dropout=.1,
                 grad_checkpoint=False):
        super(MultiheadAttentionSerial, self).__init__()
        if input_size is None:
            input_size = attention_size
        self.head_size = int(attention_size / num_heads)
        self.heads = [
            Attention(input_size,
                      self.head_size,
                      dropout=dropout,
                      grad_checkpoint=grad_checkpoint)
            for _ in num_heads]

    def forward(self, x, queries=None, values=None, attention_mask=None):
        """
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        """
        # list of (B, L, A / H)
        out = [h(x,
                 queries=queries,
                 values=values,
                 attention_mask=attention_mask)
               for h in self.heads]

        # (B, L, A)
        out = torch.cat(out, dim=-1)
        return out


class MultiheadAttentionParallel(nn.Module):
    def __init__(self,
                 attention_size=512,
                 num_heads=8,
                 input_size=None,
                 dropout=.1,
                 grad_checkpoint=False):
        super(MultiheadAttentionParallel, self).__init__()
        if input_size is None:
            input_size = attention_size
        self.dk = input_size
        self.num_heads = num_heads
        self.head_size = int(attention_size / num_heads)
        self.attention_size = attention_size
        self.grad_checkpoint = grad_checkpoint
        self.k = nn.Linear(input_size, attention_size, bias=False)
        self.q = nn.Linear(input_size, attention_size, bias=False)
        self.v = nn.Linear(input_size, attention_size, bias=False)
        self.output = FF(attention_size,
                         attention_size,
                         activation='none',
                         layer_norm=True,
                         dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self._reset_parameters()

    def _split_heads(self, x):
        """
        x => (B, L, A)
        out => (B, H, L, A/H)
        """
        batch_size, max_length, _ = x.size()
        return (x
                .view(batch_size, max_length,
                      self.num_heads, self.head_size)
                .permute(0, 2, 1, 3))

    def _merge_heads(self, x):
        """
        x => (B, H, L, A/H)
        out => (B, L, A)
        """
        batch_size, _, max_length, _ = x.size()
        # x => (B, L, H, A/H)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, max_length, -1)

    def forward(self, x, queries=None, values=None, attention_mask=None):
        """
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        """
        if queries is None:
            queries = x
        if values is None:
            values = x
        k = self._split_heads(self.k(x))  # (B, H, L, A/H)
        q = self._split_heads(self.q(queries))  # (B, H, L, A/H)
        v = self._split_heads(self.v(values))  # (B, H, L, A/H)

        # scores => (B, H, L, L)
        if self.grad_checkpoint:
            scores = checkpoint(calc_scores(self.dk), q, k)
        else:
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)
        if attention_mask is not None:
            scores = scores + ((1 - attention_mask.unsqueeze(1)) * -1e5)
        scores = F.softmax(scores, dim=-1)
        scores = self.drop(scores)

        # out => (B, H, L, A/H)
        out = self._merge_heads(torch.matmul(scores, v))
        out = self.output(out)
        return out

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.output.fc.weight)
        nn.init.constant_(self.output.fc.bias, 0.)


MultiheadAttention = MultiheadAttentionParallel


# Luong attention
class LuongAttn(nn.Module):
    def __init__(self, method, hidden_size, device='cpu'):
        super(LuongAttn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.device = device

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat(
            (hidden.expand(-1, encoder_output.size(1), -1), encoder_output),
            2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        else:
            raise NotImplementedError
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnLayer(nn.Module):
    def __init__(self, method, hidden_size, device='cpu'):
        """
        :param method: luong attention method to be implemented
        :param hidden_size: hidden size of decoder
        :param device:

        This class implements luong attention using LuongAttn class!
        Forward receives the decoder output (last hidden state at timestep
        t) and encoder output (all timesteps) and returns the output having
        applied attention and the attn weights!
        """
        super(LuongAttnLayer, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.luongattn = LuongAttn(method, hidden_size, device)
        self.out_linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, hidden, encoder_outputs):
        attn_weights = self.luongattn(hidden, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs)
        concat_input = torch.cat((hidden.squeeze(1),
                                  context.squeeze(1)), 1)
        out = torch.tanh(self.out_linear(concat_input))
        out = out.unsqueeze(1)
        return out, attn_weights
