import random
import torch
import torch.nn as nn
from slp.modules.feedforward import FF


class EmoClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmoClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.ff_layer = FF(self.input_size,self.num_classes,activation='relu')

    def forward(self, x):
        logits = self.ff_layer(x)
        return logits


class EmoSeq2SeqMultitask(nn.Module):

    def __init__(self, encoder, decoder,classifier,sos_index,device,
                 shared_emb=False):
        super(EmoSeq2SeqMultitask, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        if shared_emb:
            self.encoder.embed_in = self.decoder.embed_in
        self.sos_index = sos_index
        self.device = device

    def forward(self, inputs, input_lengths, target, target_lengths):
        encoutput, hidden = self.encoder(inputs,input_lengths)
        #last_hidden = self.encoder.get_last_layer_hidden(hidden)
        dec_init_hidden = hidden[:self.decoder.num_layers]

        # dec_init_hidden = last_hidden.view(self.decoder.num_layers,
        #                                    target.shape[0],self.decoder.hidden_size)

        decoder_input = torch.tensor([self.sos_index for _ in range(
            target.shape[0])]).long().unsqueeze(dim=1)
        decoder_input = decoder_input.to(self.device)

        if not self.decoder.attention:
            dec_out = self.decoder(decoder_input, target,
                                   dec_hidden=dec_init_hidden)
        else:
            dec_out = self.decoder(decoder_input, target,
                                   dec_hidden=dec_init_hidden,
                                   enc_output=encoutput)

        pred_emo1 = self.classifier(hidden).squeeze(0)
        return dec_out, pred_emo1
