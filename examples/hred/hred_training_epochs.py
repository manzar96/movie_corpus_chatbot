import os
import torch
import torch.nn as nn
import argparse
from torch.optim import Adam
from ignite.metrics import Loss

from slp.config.special_tokens import HRED_SPECIAL_TOKENS
from slp.data.utils import train_test_split
from slp.data.transforms import DialogSpacyTokenizer, ToTokenIds, ToTensor
from slp.data.collators import HRED_Collator
from slp.util.embeddings import EmbeddingsLoader, create_emb_file

from slp.modules.loss import SequenceCrossEntropyLoss, Perplexity
from slp.modules.seq2seq.hredseq2seq import HREDSeq2Seq
from slp.trainer.trainer import HREDTrainer,HREDTrainerEpochs

from slp.data.Semaine import SemaineDatasetTriplesOnly
from slp.data.moviecorpus import MovieCorpusDatasetTriples
from slp.data.DailyDialog import DailyDialogDataset


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
BATCH_TRAIN_SIZE = 100
BATCH_VAL_SIZE = 60



def trainer_factory(options, emb_dim, vocab_size, embeddings, pad_index,
                    sos_index, checkpoint_dir=None, device=DEVICE):

    model = HREDSeq2Seq(options, emb_dim, vocab_size, embeddings, embeddings,
                 sos_index, device)
    # init model params (rnn as orthogonal, others as normal except embeddings)
    model.init_param(model)
    numparams = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('Trainable Parameters: {}'.format(numparams))
    import ipdb;ipdb.set_trace()
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-6)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_index,reduction='sum')

    # perplexity = Perplexity(pad_index)

    # metrics = {
    #     'loss': Loss(criterion),
    #     'ppl': Loss(perplexity)}

    # trainer = HREDTrainer(model, optimizer,
    #                       checkpoint_dir=checkpoint_dir, metrics=metrics,
    #                       non_blocking=True, retain_graph=False,
    #                       patience=5,
    #                       device=device, loss_fn=criterion)

    trainer = HREDTrainerEpochs(model, optimizer, criterion,patience=5,
                                    checkpoint_dir=checkpoint_dir,
                                    device=device)
    return trainer


if __name__ == '__main__':
    # --- fix argument parser default values --
    parser = argparse.ArgumentParser(description='Main options')
    parser.add_argument('-dataset', type=str, help='Dataset used')
    parser.add_argument('-preprocess', action='store_true', default=False,
                        help='Preprocess dataset used')
    parser.add_argument('-ckpt', type=str, help='Model checkpoint')
    parser.add_argument('-embeddings', type=str, help='Embeddings file')
    parser.add_argument('-emb_dim', type=int, help='Embeddings dimension')

    parser.add_argument('-enchidden', dest='enc_hidden_size', type=int,
                        default=256, help='encoder hidden size')
    parser.add_argument('-embdrop', dest='embeddings_dropout', type=float,
                        default=0, help='embeddings dropout')
    parser.add_argument('-encembtrain', dest='enc_finetune_embeddings',
                        action='store_true', default=False,
                        help='encoder finetune embeddings')
    parser.add_argument('-encnumlayers', dest='enc_num_layers', type=int,
                        default=1, help='encoder number of layers')
    parser.add_argument('-encbi', dest='enc_bidirectional',
                        action='store_true',
                        default=False, help='bidirectional enc')
    parser.add_argument('-encdrop', dest='enc_dropout', type=float,
                        default=0, help='encoder dropout')

    parser.add_argument('-continputsize', dest='contenc_input_size',
                        type=int,
                        default=256, help='context encoder input size')
    parser.add_argument('-conthiddensize', dest='contenc_hidden_size',
                        type=int,
                        default=256, help='context encoder hidden size')
    parser.add_argument('-contnumlayers', dest='contenc_num_layers',
                        type=int,
                        default=1, help='context encoder number of layers')
    parser.add_argument('-contencdrop', dest='contenc_dropout',
                        type=float,
                        default=0, help='context encoder dropout')
    parser.add_argument('-contencbi', dest='contenc_bidirectional',
                        action='store_true',
                        default=False, help='bidirectional enc')
    parser.add_argument('-contenctype', dest='contenc_rnn_type',
                        action='store_true',
                        default='gru', help='bidirectional enc')

    parser.add_argument('-dechidden', dest='dec_hidden_size',
                        type=int,
                        default=256, help='decoder hidden size')
    parser.add_argument('-decembtrain', dest='dec_finetune_embeddings',
                        action='store_true',
                        default=False, help='decoder finetune embeddings')
    parser.add_argument('-decnumlayers', dest='dec_num_layers',
                        type=int,
                        default=1, help='decoder number of layers')
    parser.add_argument('-decbi', dest='dec_bidirectional',
                        action='store_true',
                        default=False, help='bidirectional decoder')
    parser.add_argument('-decdrop', dest='dec_dropout', type=float,
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
    parser.add_argument('-pt', dest='pretraining',
                        action='store_true',
                        default=False, help='Pretraining model (only encoder'
                                            'decoder)')
    parser.add_argument('-epochs', type=int,
                        default=1000, help='iterations for training in mini '
                                           'batches')

    parser.add_argument('-shared', action='store_true',
                        default=False, help='shared weights between encoder '
                                            'and decoder')
    parser.add_argument('-shared_emb', action='store_true',
                        default=False, help='shared embedding layer')

    options = parser.parse_args()

    # ---  read data to create vocabulary dict ---

    tokenizer = DialogSpacyTokenizer(lower=True,
                                     specials=HRED_SPECIAL_TOKENS)

    if options.dataset == "movie":
        dataset = MovieCorpusDatasetTriples('./data/', transforms=None)
    elif options.dataset == "dailydialog":
        dataset = DailyDialogDataset('./data/ijcnlp_dailydialog',
                                     transforms=None)
    elif options.dataset == "semaine":
        dataset = SemaineDatasetTriplesOnly(
        "./data/semaine-database_download_2020-01-21_11_41_49")
    else:
        assert False, "Specify dataset used in options (movie, dailydialog or" \
                      "semaine)"

    dataset.normalize_data()
    if options.preprocess:
        dataset.threshold_data(30, tokenizer=tokenizer)
        dataset.trim_words(3, tokenizer=tokenizer)
    vocab_dict = dataset.create_vocab_dict(tokenizer)

    # --- create new embedding file and load embeddings---
    new_emb_file = './cache/new_embs.txt'
    old_emb_file = options.embeddings
    freq_words_file = './cache/freq_words.txt'
    emb_dim = options.emb_dim
    create_emb_file(new_emb_file, old_emb_file, freq_words_file, vocab_dict,
                    most_freq=10000)
    word2idx, idx2word, embeddings = EmbeddingsLoader(new_emb_file, emb_dim,
                                                      extra_tokens=
                                                      HRED_SPECIAL_TOKENS
                                                      ).load()
    vocab_size = len(word2idx)
    print("Vocabulary size: {}".format(vocab_size))

    # --- set dataset transforms ---
    tokenizer = DialogSpacyTokenizer(lower=True, prepend_sos=True,
                                     append_eos=True,
                                     specials=HRED_SPECIAL_TOKENS)
    to_token_ids = ToTokenIds(word2idx, specials=HRED_SPECIAL_TOKENS)
    to_tensor = ToTensor()
    dataset = dataset.map(tokenizer).map(to_token_ids).map(to_tensor)
    print("Dataset size: {}".format(len(dataset)))
    import ipdb;ipdb.set_trace()
    # --- make train and val loaders ---

    collator_fn = HRED_Collator(device='cpu')
    train_loader, val_loader = train_test_split(dataset,
                                                batch_train=BATCH_TRAIN_SIZE,
                                                batch_val=BATCH_VAL_SIZE,
                                                collator_fn=collator_fn,
                                                test_size=0.2)

    pad_index = word2idx[HRED_SPECIAL_TOKENS.PAD.value]
    sos_index = word2idx[HRED_SPECIAL_TOKENS.SOS.value]
    eos_index = word2idx[HRED_SPECIAL_TOKENS.EOS.value]
    unk_index = word2idx[HRED_SPECIAL_TOKENS.UNK.value]
    print("sos index {}".format(sos_index))
    print("eos index {}".format(eos_index))
    print("pad index {}".format(pad_index))
    print("unk index {}".format(unk_index))

    # --- make model and train it ---

    checkpoint_dir = options.ckpt
    info_dir = os.path.join(checkpoint_dir, "info.txt")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(info_dir, "w") as info:
        info.write("DATA USED INFO\n")
        info.write("Data samples: {} \n".format(len(dataset)))
        info.write("Vocabulary size: {} \n".format(vocab_size))

        info.write("MODEL's INFO\n")
        info.write("Utterance Encoder: {} layers, {} bidirectional, "
                   "{} dropout , "
                   "{} hidden \n".format(options.enc_num_layers,
                                         options.enc_bidirectional,
                                         options.enc_dropout,
                                         options.enc_hidden_size))

        info.write("Context Encoder: {} layers, {} bidirectional, {} dropout , "
                   "{} hidden , {} input \n".format(options.contenc_num_layers,
                                                    options.contenc_bidirectional,
                                                    options.contenc_dropout,
                                                    options.contenc_hidden_size,
                                                    options.contenc_input_size))

        info.write("Decoder: {} layers, {} bidirectional, {} dropout , "
                   "{} hidden \n".format(options.dec_num_layers,
                                         options.dec_bidirectional,
                                         options.dec_dropout,
                                         options.dec_hidden_size))

        info.write("More info: \n")

        info.close()
    output_dir = os.path.join(checkpoint_dir, "output.txt")
    with open(output_dir, "w") as out:
        out.write("TRAINING AND VALIDATION RESULTS:\n")
    out.close()
    trainer = trainer_factory(options, emb_dim, vocab_size, embeddings,
                              pad_index, sos_index, checkpoint_dir,
                              device=DEVICE)

    trainer.fit(train_loader, val_loader, epochs=options.epochs)
    print("data stored in: {}\n".format(checkpoint_dir))
