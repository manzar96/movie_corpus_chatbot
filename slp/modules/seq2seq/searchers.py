import torch
import torch.nn as nn
import torch.nn.functional as F

'''HRED Seq2Seq searcher greedy using triples!'''
class GreedySearchHREDSeq2Seq(nn.Module):
    def __init__(self, hred, device):
        super(GreedySearchHREDSeq2Seq, self).__init__()
        self.enc = hred.enc
        self.cont_enc = hred.cont_enc
        self.dec = GreedySearchHREDDecoder(hred.dec, device)
        self.batch_first = hred.batch_first
        self.options = hred.options
        self.sos_index = hred.sos_index
        # we use a linear layer and tanh act function to initialize the
        # hidden of the decoder.
        # paper reference: A Hierarchical Recurrent Encoder-Decoder
        # for Generative Context-Aware Query Suggestion, 2015
        # dm,0 = tanh(D0sm−1 + b0)  (equation 7)
        self.cont_enc_to_dec = hred.cont_enc_to_dec
        self.tanh = hred.tanh
        self.device = device

    def forward(self, input_seq1, input_length1, input_seq2, input_length2):
        _, hidden1 = self.enc(input_seq1, input_length1)
        _, hidden2 = self.enc(input_seq2, input_length2)
        """
               we take the last layer of the hidden state!
               (Supposing it is a gru)
        """
        if self.options.enc_bidirectional:
            hidden1 = hidden1[-2:]
            hidden2 = hidden2[-2:]
        else:
            hidden1 = hidden1[-1]
            hidden2 = hidden2[-1]
        hidden1 = hidden1.unsqueeze(dim=1)
        hidden2 = hidden2.unsqueeze(dim=1)
        context_input = torch.cat((hidden1, hidden2), dim=1)
        _, contenc_hidden = self.cont_enc(context_input)

        dec_init_hidden = self.tanh(self.cont_enc_to_dec(contenc_hidden))
        dec_init_hidden = dec_init_hidden.view(self.options.dec_num_layers,
                                               1,
                                               self.options.dec_hidden_size)
        decoder_input = torch.tensor([self.sos_index]).long().unsqueeze(dim=1)
        decoder_input = decoder_input.to(self.device)
        dec_tokens, dec_scores = self.dec(decoder_input, dec_init_hidden)
        return dec_tokens, dec_scores


class GreedySearchHREDDecoder(nn.Module):
    def __init__(self, decoder, device):
        super(GreedySearchHREDDecoder, self).__init__()

        self.dec = decoder
        self.max_length = 11
        self.device = device

    def forward(self, dec_input, dec_hidden=None, enc_output=None):
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
        context_encoded = dec_hidden
        decoder_outputs = []

        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)

        for i in range(0, self.max_length):

            input_embed = self.dec.embed_in(dec_input)
            if enc_output is None:
                dec_out, dec_hidden = self.dec.rnn(input_embed,
                                                   hx=dec_hidden)
            else:
                assert False, "Attention is not implemented"

            # ω(dm,n−1, wm,n−1) = Ho dm,n−1 + Eo wm,n−1 + bo   (olo auto se
            # diastasi emb_size*2
            # emb_inf_vec = self.dec.emb_to_emb2(input_embed).squeeze(dim=1)
            # dec_inf_vec = self.dec.dec_to_emb2(dec_out).squeeze(dim=1)
            # cont_inf_vec = self.dec.cont_to_emb2(context_encoded).squeeze(
            #     dim=0)
            #
            # total_out = dec_inf_vec + cont_inf_vec + emb_inf_vec

            #after max_out total_out dims:  emb_size
            # total_out = self.dec.max_out(total_out)
            # out = self.dec.embed_out(total_out)

            out = self.dec.embed_out(dec_out.squeeze(dim=1))
            out = F.softmax(out, dim=1)

            decoder_scores, dec_input = torch.max(out, dim=1)
            all_tokens = torch.cat((all_tokens, dec_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            dec_input = torch.unsqueeze(dec_input, 0)

        return all_tokens, all_scores


""" Seq2Seq searcher greedy using triples but first empty"""
class GreedySearchSeq2Seq(nn.Module):
    def __init__(self, hred, device):
        super(GreedySearchSeq2Seq, self).__init__()
        self.enc = hred.enc
        self.dec = GreedySearchDecoder(hred.dec, device)
        self.batch_first = hred.batch_first
        self.options = hred.options
        self.sos_index = hred.sos_index
        self.tanh = hred.tanh
        self.device = device

    def forward(self, input_seq1, input_length1, input_seq2, input_length2):
        _, hidden1 = self.enc(input_seq2, input_length2)
        """
               we take the last layer of the hidden state!
               (Supposing it is a gru)
        """
        if self.options.enc_bidirectional:
            hidden1 = hidden1[-2:]
        else:
            hidden1 = hidden1[-1]
        hidden1 = hidden1.unsqueeze(dim=1)
        dec_init_hidden = hidden1
        dec_init_hidden = dec_init_hidden.view(self.options.dec_num_layers,
                                               1,
                                               self.options.dec_hidden_size)
        decoder_input = torch.tensor([self.sos_index]).long().unsqueeze(dim=1)
        decoder_input = decoder_input.to(self.device)
        dec_tokens, dec_scores = self.dec(decoder_input, dec_init_hidden)
        return dec_tokens, dec_scores


class GreedySearchDecoder(nn.Module):
    def __init__(self, decoder, device):
        super(GreedySearchDecoder, self).__init__()

        self.dec = decoder
        self.max_length = 11
        self.device = device

    def forward(self, dec_input, dec_hidden=None, enc_output=None):
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
        context_encoded = dec_hidden
        decoder_outputs = []

        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)

        for i in range(0, self.max_length):

            input_embed = self.dec.embed_in(dec_input)
            if enc_output is None:
                dec_out, dec_hidden = self.dec.rnn(input_embed,
                                                   hx=dec_hidden)
            else:
                assert False, "Attention is not implemented"

            out = self.dec.embed_out(dec_out.squeeze(dim=1))
            out = F.softmax(out, dim=1)

            decoder_scores, dec_input = torch.max(out, dim=1)
            all_tokens = torch.cat((all_tokens, dec_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            dec_input = torch.unsqueeze(dec_input, 0)

        return all_tokens, all_scores

