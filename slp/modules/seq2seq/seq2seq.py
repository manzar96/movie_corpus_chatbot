import random
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from slp.modules.embed import Embed
from slp.modules.attention import LuongAttnLayer
from slp.modules.treesearch import _treesearch_factory
from slp.modules.beamsearch import BeamNode, BeamGraph

def _transpose_hidden_state(hidden_state):
    """
    Transpose the hidden state so that batch is the first dimension.

    RNN modules produce (num_layers x batchsize x dim) hidden state, but DataParallel
    expects batch size to be first. This helper is used to ensure that we're always
    outputting batch-first, in case DataParallel tries to stitch things back together.
    """
    if isinstance(hidden_state, tuple):
        return tuple(map(_transpose_hidden_state, hidden_state))
    elif torch.is_tensor(hidden_state):
        return hidden_state.transpose(0, 1)
    else:
        raise ValueError("Don't know how to transpose {}".format(hidden_state))

class Encoder(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, embeddings=None,
                 embeddings_dropout=.1,
                 finetune_embeddings=False, num_layers=1, batch_first=True,
                 bidirectional=False, dropout=0, rnn_type='gru', merge_bi='sum',
                 attention=None, device='cpu'):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = input_size
        self.hidden_size = hidden_size
        self.embeddings = embeddings
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
        batch_size = inputs.shape[0]
        input_embed = self.embed_in(inputs)
        attn_mask = inputs.ne(0)
        x_emb = nn.utils.rnn.pack_padded_sequence(input_embed, lengths,
                                                  batch_first=True,
                                                  enforce_sorted=False)
        out_packed, hidden = self.rnn(x_emb)
        enc_out, out_lengths = nn.utils.rnn.pad_packed_sequence(out_packed,
                                                           batch_first=True)

        if self.bidirectional:
            # project to decoder dimension by taking sum of forward and back
            if self.rnn_type == "lstm":
                hidden = (
                    hidden[0].view(-1, self.dirs, batch_size, self.hsz).sum(1),
                    hidden[1].view(-1, self.dirs, batch_size, self.hsz).sum(1),
                )
            else:
                hidden = hidden.view(-1, self.directions, batch_size,
                                     self.hidden_size).sum(1)

            enc_out = enc_out[:, :, :self.hidden_size] + \
                         enc_out[:, :, self.hidden_size:]

        return enc_out, hidden,attn_mask

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

    def __init__(self, vocab_size, emb_size, hidden_size,start_idx,
                 embeddings=None, bidirectional=False,
                 embeddings_dropout=.1, finetune_embeddings=False,
                 num_layers=1, batch_first=True,
                 dropout=0, attention=False,
                 rnn_type="gru",
                 device='cpu'):
        """

        :param vocab_size:
        :param emb_size:
        :param hidden_size:
        :param embeddings:
        :param bidirectional:
        :param embeddings_dropout:
        :param finetune_embeddings: train embeddings
        :param num_layers:
        :param batch_first:
        :param dropout:
        :param attention:
        :param shared_weight: share weights between input embedding layer and
        output layer.
        :param rnn_type:
        :param device:
        """
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.embeddings_dropout = embeddings_dropout
        self.finetune_embeddings = finetune_embeddings
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        # This attention is being used on each time step of decoder!!!!
        # TODO: na dw gia tis alles methodous attention pws tha tis valw opws
        #  self attention!!
        self.attention = attention
        if self.attention:
            self.attn_layer = LuongAttnLayer(method='dot',
                                             hidden_size=self.hidden_size)
        self.rnn_type = rnn_type
        self.start_idx = start_idx
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

        self.hidtoemb = nn.Linear(self.hidden_size, self.emb_size, bias=True)
        # embedding to vocab size

        weight = nn.Parameter( torch.Tensor(self.vocab_size,self.emb_size).normal_(0, 1))
        rng = 1.0 / math.sqrt(self.vocab_size)
        bias = nn.Parameter(torch.Tensor(self.vocab_size).uniform_(-rng, rng))
        self.embtovoc = nn.Linear(self.emb_size,self.vocab_size)
        self.embtovoc.weight = weight
        self.embtovoc.bias = bias

    def forward(self, dec_input, enc_output=None, incremental_state=None):
        """
        Decode from input tokens.
        SOS: read incremental_state!!

        :param xs: (bsz x seqlen) LongTensor of input token indices
        :param encoder_output: output from RNNEncoder. Tuple containing
            (enc_out, enc_hidden, attn_mask) tuple.
        :param incremental_state: most recent hidden state to the decoder.
            If None, the hidden state of the encoder is used as initial state,
            and the full sequence is computed. If not None, computes only the
            next forward in the sequence.

        :returns: (output, hidden_state) pair from the RNN.

            - output is a bsz x time x latentdim matrix. If incremental_state is
                given, the time dimension will be 1. This value must be passed to
                the model's OutputLayer for a final softmax.
            - hidden_state depends on the choice of RNN
        """
        if len(enc_output) == 2:
            # attention mask should not be given if we dont use attention
            enc_state, enc_hidden = enc_output
            attn_mask = None
        else:
            enc_state, enc_hidden, attn_mask = enc_output

        attn_params = enc_state, attn_mask

        if incremental_state is not None:
            #if incremental state is not None we decode just one word! so we
            # give as input the last word of the dec_input!
            hidden = incremental_state
            dec_input = dec_input[:, -1:]

        else:
            hidden = enc_hidden

        seq_len = dec_input.shape[1]
        embed_input = self.embed_in(dec_input)

        if not self.attention:
            # no attn, we can just trust the rnn to run through
            output, new_hidden = self.rnn(embed_input, hidden)

        else:
            decoder_outputs = []
            new_hidden = hidden
            for i in range(0, seq_len):
                dec_out, new_hidden = self.rnn(embed_input[:,i,:].unsqueeze(1),
                                               hx=new_hidden)
                out, _ = self.attn_layer(dec_out, new_hidden, attn_params)
                decoder_outputs.append(out)

            output = torch.cat(decoder_outputs, dim=1).to(self.device)

        e = self.hidtoemb(output)
        scores = self.embtovoc(e)
        # set scores for padding -inf!
        scores[:, :, 0] = -1e20

        return scores, new_hidden

    def decode_full_tc(self, encoder_states, targets):
        """
        Greedy Decoding using full teacher forcing
        Used only for training and validation!!
        :param encoder_states:
        :param targets:
        :return:
        """
        # we decode given all targets as input! (full teacher-forcing)
        # TODO: see decode forced! append to start index sto target!!
        bsz = targets.size(0)
        seqlen = targets.size(1)
        inputs = targets.narrow(1, 0, seqlen - 1)
        start_inputs = torch.tensor([self.start_idx for _ in range(
            bsz)]).unsqueeze(1).to(self.device)
        inputs = torch.cat([start_inputs, inputs], 1)
        logits, _ = self.forward(inputs, encoder_states)
        _, preds = logits.max(dim=2)
        return logits, preds

    def decode_tc(self, encoder_states, targets):
        """
        Greedy Decoding using teacher forcing with ratio
        Used only for training and validation!!
        :param encoder_states:
        :param targets:
        :param tc_ratio:
        :return:
        """
        batchsize = targets.shape[0]
        seq_len = targets.shape[1]
        incr_state = None

        # fix first input!
        dec_input = [[self.start_idx for _ in range(batchsize)]]
        dec_input = torch.tensor(dec_input).to(self.device)
        all_outputs = []
        for t in range(seq_len):
            scores, hidden = self.forward(dec_input, encoder_states, incr_state)
            incr_state = hidden
            all_outputs.append(scores)
            use_teacher_forcing = True if (
                    random.random() < self.teacher_forcing_ratio) else False
            if use_teacher_forcing:
                dec_input = targets[:, t, :]
            else:
                top_index = F.log_softmax(scores, dim=1)
                _, topi = top_index.topk(1, dim=-1)
                dec_input = topi.to(self.device)

        # TODO: twra epistrefw preds kai scores !!!

    def set_tc_ratio(self, num):
        self.teacher_forcing_ratio = num

    def get_tc_ratio(self):
        return self.teacher_forcing_ratio


class BeamDecoder(nn.Module):
    """
    This version of Decoder has an implemented version for inference time
    """
    def __init__(self, vocab_size, emb_size, hidden_size,
                 embeddings=None, bidirectional=False,
                 embeddings_dropout=.1, finetune_embeddings=False,
                 num_layers=1, tc_ratio=1., batch_first=True,
                 dropout=0, attention=False, self_attention=False,
                 merge_bi='sum', rnn_type="gru", max_seq_len=15,
                 device='cpu'):
        super(BeamDecoder, self).__init__()

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
            self.attn_layer = LuongAttnLayer(method='dot',
                                             hidden_size=self.hidden_size)
        self._merge_bi = merge_bi
        self.rnn_type = rnn_type
        self.max_seq_len = max_seq_len
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
        decoder_outputs = []
        for i in range(0, targets.shape[1]):
            use_teacher_forcing = True if (
                    random.random() < self.teacher_forcing_ratio) else False
            input_embed = self.embed_in(dec_input)
            if not self.attention:
                dec_out, dec_hidden = self.rnn(input_embed, hx=dec_hidden)
                out = dec_out
            else:
                dec_out, dec_hidden = self.rnn(input_embed, hx=dec_hidden)
                out, _ = self.attn_layer(dec_out, enc_output)
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
'''
    def forward_beamdecode(self, dec_input, eos_token, sos_token,
                           dec_hidden=None, enc_output=None,
                           beam_size=1, N_best=5):

        if dec_hidden is None:
            raise NotImplementedError
        beamgraph = BeamGraph(beam_size, eos_token, sos_token)
        startnode = beamgraph.create_node(dec_hidden)
        beamgraph.prev_top_nodes.append(startnode)

        for i in range(0, self.max_seq_len):
            for node in beamgraph.prev_top_nodes:

                dec_input = torch.tensor([node.last_idx]).long().unsqueeze(
                    dim=1)
                dec_input = dec_input.to(self.device)
                dec_input = self.embed_in(dec_input)
                if not self.attention:
                    dec_out, dec_hidden = self.rnn(dec_input,
                                                   node.decoder_hidden)
                    out = dec_out
                else:
                    dec_out, dec_hidden = self.rnn(dec_input, node.dec_hidden)
                    out, _ = self.attn_layer(dec_out, enc_output)

                # apply MMI anti-language model
                # if len(sentence.sentence_idxes) < threshold:
                #     LM_output = conProb(
                #         [int(idx) for idx in sentence.sentence_idxes])
                #     decoder_output -= lamda * LM_output.view(1, 1, -1)

                out = self.embed_out(out)
                topv, topi = out.topk(beam_size)
                beamgraph.addTopk(topi, topv, dec_hidden, node)

        beamgraph.terminal_nodes += [beamgraph.toWordScore(node) for node in
                                     beamgraph.prev_top_nodes]

        beamgraph.terminal_nodes.sort(key=lambda x: x[1], reverse=True)

        terminal_nodes = beamgraph.terminal_nodes
        del beamgraph
        n = min(len(terminal_nodes), N_best)  # N-best list
        return terminal_nodes[:n]

    def set_tc_ratio(self, num):
        self.teacher_forcing_ratio = num

    def get_tc_ratio(self):
        return self.teacher_forcing_ratio

'''


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, sos_index, eos_index, device,
                 shared_emb=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if shared_emb:
            self.encoder.embed_in = self.decoder.embed_in
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.device = device

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        if torch.is_tensor(incremental_state):
            # gru or vanilla rnn
            return torch.index_select(incremental_state, 1, inds).contiguous()
        elif isinstance(incremental_state, tuple):
            return tuple(
                self.reorder_decoder_incremental_state(x, inds)
                for x in incremental_state
            )

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.
        """
        enc_out, hidden, attn_mask = encoder_states

        # LSTM or GRU/RNN hidden state?
        if isinstance(hidden, torch.Tensor):
            hid, cell = hidden, None
        else:
            hid, cell = hidden

        if not torch.is_tensor(indices):
            # cast indices to a tensor if needed
            indices = torch.LongTensor(indices).to(hid.device)

        hid = hid.index_select(1, indices)
        if cell is None:
            hidden = hid
        else:
            cell = cell.index_select(1, indices)
            hidden = (hid, cell)

        if self.decoder.attention:
            enc_out = enc_out.index_select(0, indices)
            attn_mask = attn_mask.index_select(0, indices)

        return enc_out, hidden, attn_mask


    def forward(self, inputs, input_lengths, targets, target_lengths):
        enc_output, enc_hidden, attn_mask = self.encoder(inputs, input_lengths)
        encoder_states = enc_output, enc_hidden, attn_mask
        logits, preds = self.decoder.decode_full_tc(encoder_states, targets)
        return logits

    def generate(self, inputs, input_lengths, genoptions, sos_index,
                 end_index, pad_idx):
        beam_size = genoptions.beam_size
        max_seq_len = genoptions.maxlen
        temperature = genoptions.temperature
        N_best = genoptions.N_best

        batchsize = inputs.shape[0]
        encoder_states = self.encoder(inputs, input_lengths)
        # TODO: fix set_context properly!! prepei na dinw san orisma input[
        #  batch_idx] opou to input tha einai xwris padding!!!

        beams = [_treesearch_factory(genoptions, sos_index, end_index, pad_idx,
                                     device=self.device).set_context(
            inputs[batch_idx]) for batch_idx in range(batchsize)]

        # make a an input with sos indexes [Batchsize*Beamsize , 1]
        decoder_input = (
            torch.LongTensor([sos_index]).expand(batchsize * beam_size, 1).to(
                self.device)
        )
        inds = torch.arange(batchsize).to(self.device).unsqueeze(1). repeat(
            1, beam_size).view(-1)

        encoder_states = self.reorder_encoder_states(encoder_states, inds)
        incr_state = None
        for t in range(max_seq_len):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            scores, incr_state = self.decoder(decoder_input,
                                                  encoder_states, incr_state)
            scores = scores.view(batchsize, beam_size, -1)

            if temperature != 1.0:
                scores.div_(temperature)
            scores = F.log_softmax(scores, dim=-1)

            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(scores[i])
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = self.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )

            decoder_input = torch.index_select(decoder_input, 0, incr_state_inds)
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = torch.cat([decoder_input, selection], dim=-1)

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished(N_best) for b in
                                    beams]

        # if hasattr(self, '_rerank_beams'):
        #     n_best_beam_preds_scores = self._rerank_beams(
        #         batch, n_best_beam_preds_scores
        #     )

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in
                             n_best_beam_preds_scores]
        return beam_preds_scores, beams
