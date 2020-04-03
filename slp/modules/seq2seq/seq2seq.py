import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from slp.modules.embed import Embed
from slp.modules.attention import LuongAttn


class Encoder(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, embedding=None,
                 embeddings_dropout=.1,
                 finetune_embeddings=False, num_layers=1, batch_first=True,
                 bidirectional=False, dropout=0, rnn_type='gru', merge_bi='sum',
                 attention=None, device='cpu'):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = input_size
        self.hidden_size = hidden_size
        self.embeddings = embedding
        self.embeddings_dropout = embeddings_dropout
        self.finetune_embeddings = finetune_embeddings
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.directions = 2
        else:
            self.directions = 1
        self._merge_bi = merge_bi
        self.dropout = dropout
        self.attention = attention
        self.rnn_type = rnn_type
        self.device = device

        self.embed_in = Embed(num_embeddings=self.vocab_size,
                              embedding_dim=self.emb_size,
                              embeddings=self.embeddings,
                              dropout=self.embeddings_dropout,
                              trainable=self.finetune_embeddings)

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.emb_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               bidirectional=self.bidirectional,
                               dropout=self.dropout,
                               batch_first=self.batch_first)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=self.emb_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=self.bidirectional,
                              dropout=self.dropout,
                              batch_first=self.batch_first)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.emb_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=self.bidirectional,
                              dropout=self.dropout,
                              batch_first=self.batch_first)

    def forward(self, inputs, lengths):

        input_embed = self.embed_in(inputs)
        x_emb = nn.utils.rnn.pack_padded_sequence(input_embed, lengths,
                                                  batch_first=True,
                                                  enforce_sorted=False)
        out_packed, hidden = self.rnn(x_emb)
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(out_packed,
                                                           batch_first=True)

        if self.bidirectional:
            enc_out = out[:, :, :self.hidden_size] + \
                         out[:, :, self.hidden_size:]

        return enc_out, hidden

    def merge_bi(self, hidden_fwd, hidden_bwd):
        """
        :param hidden_fwd:\
        :param hidden_bwd
        if merge_bi=="sum":sum fwd and bwd hidden states of the last layer
        to return as last
        if merge_bi=="fwd": return the fwd layer
        if merge_bi=="bwd: return the bwd layer
        if merge_bi=="cat" concatenate fwd and bwd
        :return: hidden
        """
        if self._merge_bi == 'sum':
            hidden = hidden_fwd+hidden_bwd
        elif self._merge_bi == 'avg':
            hidden = (hidden_fwd+hidden_bwd)/2
        elif self._merge_bi == 'fwd':
            hidden = hidden_fwd
        elif self._merge_bi == 'bwd':
            hidden = hidden_bwd
        elif self._merge_bi == "cat":
            hidden = torch.cat((hidden_fwd, hidden_bwd), dim=-1)
        else:
            assert False, "wrong merge bi input!!"
        return hidden

    def get_last_layer_hidden(self, hidden):
        """
        Returns the last_hidden state: [1,batch_size,hidden_size]
        :param hidden: hidden state
        :return: last hidden
        """
        if self.rnn_type == 'lstm':
            batch_size = hidden[0].shape[1]
            hid, cell = hidden

            if self.bidirectional:
                hid = hid.view(self.num_layers, self.directions, batch_size,
                               self.hidden_size)
                cell = cell.view(self.num_layers, self.directions,
                                 batch_size, self.hidden_size)
                last_hid = hid[-1]
                last_hid_fwd = last_hid[0]
                last_hid_bwd = last_hid[1]
                last_hid = self.merge_bi(last_hid_fwd, last_hid_bwd)
                last_hid = last_hid.unsqueeze(0)
                last_cell = cell[-1]
                last_cell_fwd = last_cell[0]
                last_cell_bwd = last_cell[1]
                last_cell = self.merge_bi(last_cell_fwd, last_cell_bwd)
                last_cell = last_cell.unsqueeze(0)
                last_hidden = last_hid, last_cell
            else:
                last_hidden = (hid[-1].unsqueeze(0), cell[-1].unsqueeze(0))
        else:
            batch_size = hidden.shape[1]
            if self.bidirectional:
                hidden = hidden.view(self.num_layers, self.directions,
                                     batch_size, self.hidden_size)
                last_hidden = hidden[-1]
                last_hidden_fwd = last_hidden[0]
                last_hidden_bwd = last_hidden[1]
                last_hidden = self.merge_bi(last_hidden_fwd, last_hidden_bwd)
                last_hidden = last_hidden.unsqueeze(0)
            else:
                last_hidden = hidden[-1].unsqueeze(0)
        return last_hidden


class Decoder(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size,
                 embeddings=None, bidirectional=False,
                 embeddings_dropout=.1, finetune_embeddings=False,
                 num_layers=1, tc_ratio=1., batch_first=True,
                 dropout=0, attention=False, self_attention=False,
                 merge_bi='sum', rnn_type="gru",
                 device='cpu'):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.embeddings_dropout = embeddings_dropout
        self.finetune_embeddings = finetune_embeddings
        self.num_layers = num_layers
        self.teacher_forcing_ratio = tc_ratio
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        # This attention is being used on each time step of decoder!!!!
        # TODO: na dw gia tis alles methodous attention pws tha tis valw!!
        self.attention = attention
        if self.attention:
            self.concat = nn.Linear(hidden_size * 2, hidden_size)
            self.attn_layer = LuongAttn(method='dot',
                                        hidden_size=self.hidden_size)
        self._merge_bi = merge_bi
        self.rnn_type = rnn_type
        self.device = device

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.emb_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               bidirectional=self.bidirectional,
                               dropout=self.dropout,
                               batch_first=self.batch_first)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=self.emb_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=self.bidirectional,
                              dropout=self.dropout,
                              batch_first=self.batch_first)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.emb_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=self.bidirectional,
                              dropout=self.dropout,
                              batch_first=self.batch_first)

        self.embed_in = Embed(num_embeddings=self.vocab_size,
                              embedding_dim=self.emb_size,
                              embeddings=self.embeddings,
                              dropout=self.embeddings_dropout,
                              trainable=self.finetune_embeddings)

        self.embed_out = nn.Linear(self.hidden_size, self.vocab_size, False)

    def forward(self, dec_input, targets, dec_hidden=None,
                enc_output=None):
        max_seq_len = targets.shape[1]
        decoder_outputs = []
        for i in range(0, max_seq_len):
            use_teacher_forcing = True if (
                    random.random() < self.teacher_forcing_ratio) else False

            input_embed = self.embed_in(dec_input)
            if self.attention is None:
                dec_out, dec_hidden = self.rnn(input_embed, hx=dec_hidden)
                out = dec_out
            else:
                dec_out, dec_hidden = self.rnn(input_embed, hx=dec_hidden)
                attn_weights = self.attn_layer(dec_out, enc_output)
                context = attn_weights.bmm(enc_output)
                concat_input = torch.cat((dec_out.squeeze(1),
                                          context.squeeze(1)), 1)
                out = torch.tanh(self.concat(concat_input))

            out = self.embed_out(out.squeeze(dim=1))
            decoder_outputs.append(out)
            if use_teacher_forcing:
                dec_input = targets[:, i].unsqueeze(dim=1)
            else:
                dec_out = dec_out.squeeze(1)
                top_index = F.log_softmax(dec_out, dim=1)
                _, topi = top_index.topk(1, dim=-1)
                dec_input = topi.to(self.device)

        dec_output = torch.stack(decoder_outputs).transpose(0, 1).contiguous()
        return dec_output

    def set_tc_ratio(self, num):
        self.teacher_forcing_ratio = num

    def get_tc_ratio(self):
        return self.teacher_forcing_ratio

    def greedy_decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.vocab_size)
        word_probs = F.log_softmax(logits_reshape, dim=1)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,sos_index,device,shared_emb=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if shared_emb:
            self.encoder.embed_in = self.decoder.embed_in
        self.sos_index = sos_index
        self.device = device

    def forward(self, inputs, input_lengths, target, target_lengths):
        encoutput, hidden = self.encoder(inputs)
        #last_hidden = self.encoder.get_last_layer_hidden(hidden)

        dec_init_hidden = hidden.view(self.decoder.num_layers, target.shape[0],
                                               self.dec.hidden_size)
        decoder_input = torch.tensor([self.sos_index for _ in range(
            target.shape[0])]).long().unsqueeze(dim=1)
        decoder_input = decoder_input.to(self.device)

        if self.decoder.attention is None:
            dec_out = self.decoder(decoder_input, target,
                                   dec_hidden=dec_init_hidden)
        else:
            dec_out = self.decoder(decoder_input, target,
                                   dec_hidden=dec_init_hidden,
                                   enc_output=encoutput)

        return dec_out
