import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from slp.modules.beamsearch import BeamNode, BeamGraph

from slp.modules.seq2seq.seq2seq import Seq2Seq

class SearcherDecoder(nn.Module):
    def __init__(self, decoder, max_len, eos_index, sos_index, device):
        super(SearcherDecoder, self).__init__()
        self.embed_in = decoder.embed_in
        self.rnn = decoder.rnn
        self.embed_out = decoder.embed_out
        self.vocab_size = decoder.vocab_size
        self.attention = decoder.attention
        self.num_layers = decoder.num_layers
        if self.attention:
            self.attn_layer = decoder.attn_layer
        self.max_seq_len = max_len
        self.sos_index=sos_index
        self.eos_index = eos_index
        self.device = device

    def forward(self, *args, **kwargs):
        """must be overwitten"""
        raise NotImplementedError

class GreedyDecoder(SearcherDecoder):

    def forward(self, dec_input, dec_hidden=None, enc_output=None):
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        decoder_outputs = []
        for i in range(0, self.max_seq_len):
            input_embed = self.embed_in(dec_input)
            if not self.attention:
                dec_out, dec_hidden = self.rnn(input_embed, hx=dec_hidden)
                out = dec_out
            else:
                dec_out, dec_hidden = self.rnn(input_embed, hx=dec_hidden)
                out,_ = self.attn_layer(dec_out, enc_output)

            out = self.embed_out(out.squeeze(dim=1))
            decoder_outputs.append(out)
            out = F.softmax(out, dim=1)
            decoder_scores, dec_input = torch.max(out, dim=1)
            all_tokens = torch.cat((all_tokens, dec_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            if dec_input == self.eos_index:
                break
            dec_input = torch.unsqueeze(dec_input, 0)

        dec_output = torch.stack(decoder_outputs).transpose(0, 1).contiguous()
        return all_tokens, all_scores, dec_output


class BeamDecoder(SearcherDecoder):

    def forward(self, dec_input, dec_hidden=None, enc_output=None,
                beam_size=1, N_best=5):

        if dec_hidden is None:
            raise NotImplementedError
        beamgraph = BeamGraph(beam_size, self.eos_index, self.sos_index)
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
                    dec_out, dec_hidden = self.rnn(dec_input,
                                                   node.decoder_hidden)
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


class GreedySeq2Seq(nn.Module):
    def __init__(self, seq2seq, max_len, device):
        super(GreedySeq2Seq, self).__init__()
        self.encoder = seq2seq.encoder
        self.sos_index = seq2seq.sos_index
        self.eos_index = seq2seq.eos_index
        self.decoder = GreedyDecoder(seq2seq.decoder,max_len,self.eos_index,
                                     self.sos_index, device)
        self.device = device

    def forward(self, inputs, input_lengths):
        encoutput, hidden = self.encoder(inputs,input_lengths)
        #last_hidden = self.encoder.get_last_layer_hidden(hidden)
        dec_init_hidden = hidden[:self.decoder.num_layers]

        # dec_init_hidden = last_hidden.view(self.decoder.num_layers,
        #                                    target.shape[0],self.decoder.hidden_size)

        decoder_input = torch.tensor([self.sos_index]).long().unsqueeze(dim=1)
        decoder_input = decoder_input.to(self.device)

        if not self.decoder.attention:
            tokens,scores,logits = self.decoder(decoder_input,
                                         dec_hidden=dec_init_hidden)
        else:
            tokens,scores,logits = self.decoder(decoder_input,
                                         dec_hidden=dec_init_hidden,
                                         enc_output=encoutput)
        return tokens,logits


class BeamSeq2Seq(nn.Module):
    def __init__(self, seq2seq, max_len, beam_size, N_best, device):
        super(BeamSeq2Seq, self).__init__()
        self.encoder = seq2seq.encoder
        self.sos_index = seq2seq.sos_index
        self.eos_index = seq2seq.eos_index
        self.decoder = BeamDecoder(seq2seq.decoder, max_len,
                                   self.eos_index, self.sos_index, self.device)

        self.beam_size = beam_size
        self.N_best = N_best
        self.device = device

    def forward(self, inputs, input_lengths):
        encoutput, hidden = self.encoder(inputs,input_lengths)
        #last_hidden = self.encoder.get_last_layer_hidden(hidden)
        dec_init_hidden = hidden[:self.decoder.num_layers]

        # dec_init_hidden = last_hidden.view(self.decoder.num_layers,
        #                                    target.shape[0],self.decoder.hidden_size)

        decoder_input = torch.tensor([self.sos_index]).long().unsqueeze(dim=1)
        decoder_input = decoder_input.to(self.device)

        if not self.decoder.attention:
            outs = self.decoder(decoder_input, self.eos_index,
                                self.sos_index, dec_hidden=dec_init_hidden,
                                beam_size=self.beam_size, N_best=self.N_best)
        else:
            outs = self.decoder(decoder_input, self.eos_index,
                                self.sos_index, dec_hidden=dec_init_hidden,
                                enc_output=encoutput,
                                beam_size=self.beam_size, N_best=self.N_best)
        return outs
