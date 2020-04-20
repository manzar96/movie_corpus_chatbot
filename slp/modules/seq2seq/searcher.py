import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from slp.modules.seq2seq.seq2seq import Seq2Seq

class SearcherDecoder(nn.Module):

    def __init__(self, decoder,max_len,device):
        super(SearcherDecoder, self).__init__()
        self.embed_in = decoder.embed_in
        self.rnn = decoder.rnn
        self.embed_out = decoder.embed_out
        self.vocab_size = decoder.vocab_size
        self.attention = decoder.attention
        self.num_layers = decoder.num_layers
        if self.attention:
            self.attn_layer = self.decoder.attn_layer
            self.concat = self.decoder.concat
        self.max_len=max_len
        self.device = device

    def forward(self, dec_input,dec_hidden=None, enc_output=None):
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        decoder_outputs = []
        for i in range(0, self.max_len):

            input_embed = self.embed_in(dec_input)
            if not self.attention:
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
            out = F.softmax(out, dim=1)
            decoder_scores, dec_input = torch.max(out, dim=1)
            all_tokens = torch.cat((all_tokens, dec_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            dec_input = torch.unsqueeze(dec_input, 0)

        dec_output = torch.stack(decoder_outputs).transpose(0, 1).contiguous()
        return all_tokens, all_scores, dec_output



class SearcherSeq2Seq(nn.Module):
    def __init__(self, seq2seq,max_len, device):
        super(SearcherSeq2Seq, self).__init__()
        self.encoder = seq2seq.encoder
        self.decoder = SearcherDecoder(seq2seq.decoder,max_len,device)
        #check here if emb is same!!!
        self.sos_index = seq2seq.sos_index
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


class BeamSearcherSeq2Seq(nn.Module):
    def __init__(self, seq2seq, beam_size, N_best, device):
        super(BeamSearcherSeq2Seq, self).__init__()
        self.encoder = seq2seq.encoder
        self.decoder = seq2seq.decoder

        self.sos_index = seq2seq.sos_index
        self.eos_index = seq2seq.eos_index
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
            outs = self.decoder.forward_beamdecode(decoder_input,
                                                   self.eos_index,
                                                   self.sos_index,
                                                   dec_hidden=dec_init_hidden,
                                                   beam_size=self.beam_size,
                                                   N_best=self.N_best)
        return outs