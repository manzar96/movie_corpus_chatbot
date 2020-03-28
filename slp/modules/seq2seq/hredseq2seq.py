import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from slp.modules.embed import Embed
from slp.modules.pooling import L2PoolingLayer, Maxout2


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
        out_padded, out_lengths = nn.utils.rnn.pad_packed_sequence(out_packed,
                                                           batch_first=True)
        return out_padded, hidden

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


class ContextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True,
                 bidirectional=False, dropout=0, attention=None,
                 rnn_type='gru', merge_bi='sum', device='cpu'):
        super(ContextEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
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

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               bidirectional=self.bidirectional,
                               dropout=self.dropout,
                               batch_first=self.batch_first)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=self.bidirectional,
                              dropout=self.dropout,
                              batch_first=self.batch_first)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=self.bidirectional,
                              dropout=self.dropout,
                              batch_first=self.batch_first)

    def forward(self, encoded_context):
        out, hidden = self.rnn(encoded_context)
        return out, hidden

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


class HREDDecoder(nn.Module):
    """
    This implementation of the decoder is only used for the referenced paper in
    HRED class. That's because of the used of some linear layers, max-out
    methods!
    """

    def __init__(self, options, vocab_size, emb_size, hidden_size,
                 embeddings=None,
                 embeddings_dropout=.1, finetune_embeddings=False,
                 num_layers=1, tc_ratio=1., batch_first=True,
                 bidirectional=False, dropout=0, attention=None,
                 merge_bi='sum', rnn_type="gru", device='cpu', encoder=None):
        super(HREDDecoder, self).__init__()

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
        self.attention = attention
        self._merge_bi = merge_bi
        self.rnn_type = rnn_type
        self.device = device
        self.pretraining = options.pretraining

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

        self.tanh = nn.Tanh()

        '''Uncomment if you dont want to use maxout'''
        self.dec_to_emb2 = nn.Linear(self.hidden_size, self.emb_size*2, False)
        self.emb_to_emb2 = nn.Linear(self.emb_size, self.emb_size * 2, True)
        if options.pretraining:
            self.cont_to_emb2 = nn.Linear(2*options.contenc_hidden_size,
                                          self.emb_size*2, False)
        else:
            self.cont_to_emb2 = nn.Linear(options.contenc_hidden_size,
                                          self.emb_size*2, False)
        self.max_out = Maxout2(self.emb_size*2, self.emb_size, 2)

        self.embed_out = nn.Linear(self.emb_size, self.vocab_size, False)

    def forward(self, dec_input, targets, target_lens, dec_hidden=None,
                context_encoded=None, enc_output=None):
        """
        dec_hidden is used for decoder's hidden state initialization!
        Usually the encoder's last (from the last timestep) hidden state is
        passed to decoder's hidden state.
        enc_output: argument is passed if we want to have attention (it is
        used only for attention, if you don't want to have attention on your
        model leave it as is!)
        dec_lengths: during decoding lengths is not mandatory. however we
        pass this argument because word rnn receives as input the lengths of
        input too. (We cannot skip giving lengths because in another
        situations where samples are padded we want to receive the last
        unpadded element for every sample in the batch and not the one for
        t=seq_len)
        """
        max_seq_len = targets.shape[1]
        decoder_outputs = []
        for i in range(0, max_seq_len):
            use_teacher_forcing = True if (
                    random.random() < self.teacher_forcing_ratio) else False

            if use_teacher_forcing:
                input_embed = self.embed_in(dec_input)
                if enc_output is None:
                    dec_out, dec_hidden = self.rnn(input_embed, hx=dec_hidden)
                else:
                    assert False, "Attention is not implemented"

                '''Uncomment if want to use maxout!'''
                '''SOS!! you should uncomment from init too!'''
                # ω(dm,n−1, wm,n−1) = Ho dm,n−1 + Eo wm,n−1 + bo   (olo auto se
                # diastasi emb_size*2
                # emb_inf_vec = self.emb_to_emb2(input_embed).squeeze(1)
                # dec_inf_vec = self.dec_to_emb2(dec_out).squeeze(1)
                # cont_inf_vec = self.cont_to_emb2(context_encoded).squeeze(0)
                # total_out = dec_inf_vec + cont_inf_vec + emb_inf_vec
                # total_out = self.max_out(total_out)
                # out = self.embed_out(total_out)
                # decoder_outputs.append(out)

                '''Uncomment if want to use simple encdec'''
                out = self.embed_out(dec_out.squeeze(dim=1))
                decoder_outputs.append(out)

                dec_input = targets[:, i].unsqueeze(dim=1)

            else:
                input_embed = self.embed_in(dec_input)
                if enc_output is None:
                    dec_out, dec_hidden = self.rnn(input_embed, hx=dec_hidden)
                else:
                    assert False, "Attention is not implemented"

        dec_output = torch.stack(decoder_outputs).transpose(0, 1).contiguous()
        return dec_output

    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.vocab_size)
        word_probs = F.log_softmax(logits_reshape, dim=1)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs


class HREDSeq2Seq(nn.Module):
    def __init__(self, options, emb_size, vocab_size, enc_embeddings,
                 dec_embeddings, sos_index, device):
        super(HREDSeq2Seq, self).__init__()
        self.enc = Encoder(input_size=emb_size,
                           vocab_size=vocab_size,
                           embedding=enc_embeddings,
                           hidden_size=options.enc_hidden_size,
                           embeddings_dropout=options.embeddings_dropout,
                           finetune_embeddings=options.enc_finetune_embeddings,
                           num_layers=options.enc_num_layers,
                           batch_first=options.batch_first,
                           bidirectional=options.enc_bidirectional,
                           dropout=options.enc_dropout,
                           rnn_type=options.enc_rnn_type,
                           device=device)

        self.cont_enc = ContextEncoder(input_size=options.contenc_input_size,
                                       hidden_size=options.contenc_hidden_size,
                                       num_layers=options.contenc_num_layers,
                                       batch_first=options.batch_first,
                                       bidirectional=
                                       options.contenc_bidirectional,
                                       dropout=options.contenc_dropout,
                                       rnn_type=options.contenc_rnn_type,
                                       device=device)

        self.dec = HREDDecoder(options, vocab_size=vocab_size,
                               emb_size=emb_size,
                               hidden_size=options.dec_hidden_size,
                               embeddings=dec_embeddings,
                               embeddings_dropout=options.embeddings_dropout,
                               finetune_embeddings=
                               options.dec_finetune_embeddings,
                               num_layers=options.dec_num_layers,
                               tc_ratio=options.teacherforcing_ratio,
                               batch_first=options.batch_first,
                               bidirectional=options.dec_bidirectional,
                               dropout=options.dec_dropout,
                               merge_bi=options.dec_merge_bi,
                               rnn_type=options.dec_rnn_type,
                               device=device)

        if options.shared_emb:
            self.dec.embed_in = self.enc.embed_in

        if options.shared:
            self.dec.embed_in = self.enc.embed_in
            self.dec.rnn = self.enc.rnn

        self.batch_first = options.batch_first
        self.options = options
        self.sos_index = sos_index
        self.device = device

        # we use a linear layer and tanh act function to initialize the
        # hidden of the decoder.
        # paper reference: A Hierarchical Recurrent Encoder-Decoder
        # for Generative Context-Aware Query Suggestion, 2015
        # dm,0 = tanh(D0sm−1 + b0)  (equation 7)

        if self.options.pretraining:
            self.cont_enc_to_dec = nn.Linear(2*self.enc.hidden_size,
                                             self.dec.hidden_size, bias=True)
        else:
            self.cont_enc_to_dec = nn.Linear(self.cont_enc.hidden_size,
                                             self.dec.hidden_size, bias=True)
        self.tanh = nn.Tanh()

        # if self.options.pretraining:
        #     for param in self.cont_enc.rnn.parameters():
        #         if param.requires_grad:
        #             param.requires_grad = False
        #     for param in self.cont_enc_to_dec.parameters():
        #         if param.requires_grad:
        #             param.requires_grad = False
        # else:
        #     for param in self.cont_enc.rnn.parameters():
        #         param.requires_grad = True
        #     for param in self.cont_enc_to_dec.parameters():
        #         param.requires_grad = True

    def forward(self, u1, l1, u2, l2, u3, l3):
        # this one can be used if during pretraining we want to do not pass
        # utterances through context encoder!!!
        '''
        if self.options.pretraining:
            _, hidden = self.enc(u2, l2)
            hidden = self.enc.get_last_layer_hidden(hidden)
            dec_init_hidden = hidden.view(self.options.dec_num_layers,
                                          u3.shape[0],
                                          self.options.dec_hidden_size)
            #decoder_input = torch.zeros(u3.shape[0], 1).long()
            decoder_input = torch.tensor([self.sos_index for _ in range(
                u3.shape[0])]).long().unsqueeze(dim=1)
            decoder_input = decoder_input.to(self.device)
            dec_out = self.dec(decoder_input, u3, l3, dec_init_hidden)
        else:
            _, hidden1 = self.enc(u1, l1)
            _, hidden2 = self.enc(u2, l2)
            hidden1 = self.enc.get_last_layer_hidden(hidden1)
            hidden2 = self.enc.get_last_layer_hidden(hidden2)

            # TODO: try cat on second dim!
            context_input = torch.cat((hidden1, hidden2), dim=1)

            _, cont_hidden = self.cont_enc(context_input)
            dec_init_hidden = self.tanh(self.cont_enc_to_dec(cont_hidden))

            dec_init_hidden = dec_init_hidden.view(self.options.dec_num_layers,
                                                   u3.shape[0],
                                                   self.options.dec_hidden_size)

            #decoder_input = torch.zeros(u3.shape[0], 1).long()
            decoder_input = torch.tensor([self.sos_index for _ in range(
                u3.shape[0])]).long().unsqueeze(dim=1)
            decoder_input = decoder_input.to(self.device)
            dec_out = self.dec(decoder_input, u3, l3, dec_init_hidden,
                               context_encoded=None)
        return dec_out
        '''
        _, hidden1 = self.enc(u1, l1)
        _, hidden2 = self.enc(u2, l2)
        hidden1 = self.enc.get_last_layer_hidden(hidden1)
        hidden2 = self.enc.get_last_layer_hidden(hidden2)
        if self.options.pretraining:
            context_input = torch.cat((hidden1, hidden2), dim=-1)
            cont_hidden = context_input
            # we do not pass it from context encoder!!
        else:
            # TODO: try cat on dim 2 but set input size: 2*encoder hidden!
            context_input = torch.cat((hidden1, hidden2), dim=0)
            context_input = context_input.transpose(0, 1).contiguous()
            _, cont_hidden = self.cont_enc(context_input)

        dec_init_hidden = self.tanh(self.cont_enc_to_dec(cont_hidden))
        dec_init_hidden = dec_init_hidden.view(self.dec.num_layers,
                                               u3.shape[0],
                                               self.dec.hidden_size)

        #decoder_input = torch.zeros(u3.shape[0], 1).long()
        decoder_input = torch.tensor([self.sos_index for _ in range(
            u3.shape[0])]).long().unsqueeze(dim=1)
        decoder_input = decoder_input.to(self.device)
        dec_out = self.dec(decoder_input, u3, l3, dec_init_hidden,
                           context_encoded=cont_hidden)
        return dec_out

    def init_param(self, model):
        for name, param in model.named_parameters():
            # skip over the embeddings so that the padding index ones are 0
            if 'embed' in name:
                continue
            elif ('rnn' in name or 'lm' in name) and len(param.size()) >= 2:
                nn.init.orthogonal_(param)
            else:
                nn.init.normal_(param, 0, 0.01)


