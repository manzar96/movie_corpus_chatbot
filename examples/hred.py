import numpy as np
import torch
import argparse
import os
import torch.nn as nn

from ignite.metrics import Loss

from torch.optim import Adam

from slp.config.special_tokens import HRED_SPECIAL_TOKENS
from slp.data.utils import train_test_split
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.data.SubtleTriples import SubTriples2
from slp.data.Semaine import SemaineDatasetTriplesOnly
from slp.data.collators import HRED_Collator
from slp.util.embeddings import EmbeddingsLoader
from slp.trainer.trainer import HREDTrainer
from slp.modules.loss import SequenceCrossEntropyLoss
from slp.modules.seq2seq.hred import HRED

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_EPOCHS = 10
BATCH_TRAIN_SIZE = 16
BATCH_VAL_SIZE = 16


def trainer_factory(options,emb_dim,vocab_size,embeddings,  pad_index, sos_index, device=DEVICE):

    model = HRED(options, emb_dim, vocab_size, embeddings, embeddings,
                 sos_index, device)

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-6)

    criterion = SequenceCrossEntropyLoss(pad_index)

    metrics = {
        'loss': Loss(criterion)
    }

    trainer = HREDTrainer(model, optimizer, # checkpoint_dir=None,
                                      metrics=metrics, non_blocking=True,
                                      retain_graph=False, patience=5,
                                      device=device, loss_fn=criterion)
    return trainer


if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='HRED parameter options')
    parser.add_argument('-n', dest='name', help='enter suffix for model files')
    parser.add_argument('-model_path', dest='model_path', default='./models', help='enter the path in which you want to store the model state')
    parser.add_argument('-enchidden', dest='enc_hidden_size',
                        action='store_true',
                        default=512, help='encoder hidden size')
    parser.add_argument('-embdrop', dest='embeddings_dropout',
                        action='store_true',
                        default=0, help='embeddings dropout')
    parser.add_argument('-encembtrain', dest='enc_finetune_embeddings',
                        action='store_true',
                        default=False, help='encoder finetune embeddings')
    parser.add_argument('-encnumlayers', dest='enc_num_layers',
                        action='store_true',
                        default=1, help='encoder number of layers')
    parser.add_argument('-encbi', dest='enc_bidirectional',
                        action='store_true',
                        default=False, help='bidirectional enc')
    parser.add_argument('-encdrop', dest='enc_dropout', action='store_true',
                        default=0, help='encoder dropout')

    parser.add_argument('-continputsize', dest='contenc_input_size',
                        action='store_true',
                        default=512, help='context encoder input size')
    parser.add_argument('-conthiddensize', dest='contenc_hidden_size',
                        action='store_true',
                        default=512, help='context encoder hidden size')
    parser.add_argument('-contnumlayers', dest='contenc_num_layers',
                        action='store_true',
                        default=1, help='context encoder number of layers')
    parser.add_argument('-contencdrop', dest='contenc_dropout',
                        action='store_true',
                        default=0, help='context encoder dropout')
    parser.add_argument('-contencbi', dest='contenc_bidirectional',
                        action='store_true',
                        default=False, help='bidirectional enc')
    parser.add_argument('-contenctype', dest='contenc_rnn_type',
                        action='store_true',
                        default='gru', help='bidirectional enc')

    parser.add_argument('-dechidden', dest='dec_hidden_size',
                        action='store_true',
                        default=512, help='decoder hidden size')
    parser.add_argument('-decembtrain', dest='dec_finetune_embeddings',
                        action='store_true',
                        default=False, help='decoder finetune embeddings')
    parser.add_argument('-decnumlayers', dest='dec_num_layers',
                        action='store_true',
                        default=1, help='decoder number of layers')
    parser.add_argument('-decbi', dest='dec_bidirectional',
                        action='store_true',
                        default=False, help='bidirectional decoder')
    parser.add_argument('-decdrop', dest='dec_dropout', action='store_true',
                        default=0, help='decoder dropout')
    parser.add_argument('-decmergebi', dest='dec_merge_bi',
                        action='store_true',
                        default='cat', help='decoder merge bidirectional '
                                       'method')
    parser.add_argument('-dectype', dest='dec_rnn_type',
                        action='store_true',
                        default='gru', help='decoder rnn type')

    parser.add_argument('-bf', dest='batch_first', action='store_true',
                        default=True, help='batch first')
    parser.add_argument('-tf', dest='teacherforcing_ratio',
                        action='store_true',
                        default=1., help='teacher forcing ratio')

    options = parser.parse_args()
    print(options)


    emb_file = './cache/glove.6B.50d.txt'
    emb_dim = 50
    word2idx, idx2word, embeddings = EmbeddingsLoader(emb_file, emb_dim,
                                                      extra_tokens=
                                                      HRED_SPECIAL_TOKENS
                                                      ).load()
    vocab_size = len(word2idx)
    tokenizer = SpacyTokenizer(specials=HRED_SPECIAL_TOKENS)
    to_token_ids = ToTokenIds(word2idx, specials=HRED_SPECIAL_TOKENS)
    to_tensor = ToTensor()
    # dataset = SubTriples2('./data/corpus0sDialogues.txt', transforms=[
    #     tokenizer, to_token_ids, to_tensor])
    dataset = SemaineDatasetTriplesOnly(
        "./data/semaine-database_download_2020-01-21_11_41_49", transforms=[
            tokenizer, to_token_ids, to_tensor])

    collator_fn = HRED_Collator(device='cpu')
    train_loader, val_loader = train_test_split(dataset,
                                                batch_train=BATCH_TRAIN_SIZE,
                                                batch_val=BATCH_VAL_SIZE,
                                                collator_fn=collator_fn,
                                                test_size=0.2)

    pad_index = word2idx[HRED_SPECIAL_TOKENS.PAD.value]
    sos_index = word2idx[HRED_SPECIAL_TOKENS.SOU.value]
    eos_index = word2idx[HRED_SPECIAL_TOKENS.EOU.value]
    print(sos_index)

    # trainer = trainer_factory(options, emb_dim, vocab_size, embeddings,
    #                           pad_index, sos_index, device=DEVICE)
    #
    # final_score = trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS)
    # print("END")


    model = HRED(options, emb_dim, vocab_size, embeddings, embeddings,
                 sos_index, DEVICE)

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-6)

    criterion = SequenceCrossEntropyLoss(pad_index)

    model.to(DEVICE)
    clip = 1
    for epoch in range(MAX_EPOCHS):
        avg_epoch_loss = 0
        model.train()
        last = 0
        for batch_idx, batch in enumerate(train_loader):
            u1, l1, u2, l2, u3, l3 = batch
            u1 = u1.to(DEVICE)
            u2 = u2.to(DEVICE)
            u3 = u3.to(DEVICE)
            l1 = l1.to(DEVICE)
            l2 = l2.to(DEVICE)
            l3 = l3.to(DEVICE)

            if clip is not None:
                _ = nn.utils.clip_grad_norm_(model.parameters(), clip)

            output = model(u1, l1, u2, l2, u3, l3)
            loss = criterion(output, u3)
            avg_epoch_loss += loss.item()
            loss.backward(retain_graph=False)

            optimizer.step()

            last = batch_idx

        avg_epoch_loss = avg_epoch_loss / (last + 1)
        print("Epoch {} , avg_epoch_loss {}".format(epoch, avg_epoch_loss))

