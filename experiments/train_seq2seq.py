import os
import torch
import torch.nn as nn
import argparse
import pickle

from slp.data.utils import train_test_split
from slp.data.vocab import word2idx_from_dataset
from slp.util.embeddings import EmbeddingsLoader, create_emb_file
from slp.config.special_tokens import DIALOG_SPECIAL_TOKENS
from slp.data.transforms import DialogSpacyTokenizer, ToTokenIds, ToTensor
from slp.data.DailyDialog import DailyDialogDatasetTuples
from slp.data.collators import Seq2SeqCollator
from torch.optim import Adam
#from slp.modules.loss import SequenceCrossEntropyLoss
from slp.trainer.seq2seqtrainer import Seq2SeqTrainerEpochs
from slp.modules.seq2seq.seq2seq import Encoder,Decoder,Seq2Seq

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
BATCH_TRAIN_SIZE = 64
BATCH_VAL_SIZE = 64

def print_model_info(checkpoint_dir,dataset,vocab_size,options):
    info_dir = os.path.join(checkpoint_dir, "info.txt")
    with open(info_dir, "w") as info:
        info.write("DATA USED INFO\n")
        info.write("Data samples: {} \n".format(len(dataset)))
        info.write("Vocabulary size: {} \n".format(vocab_size))

        info.write("MODEL's INFO\n")
        info.write("Encoder: {} layers, {} bidirectional, "
                   "{} dropout , "
                   "{} hidden \n".format(options.enc_num_layers,
                                         options.enc_bidirectional,
                                         options.enc_dropout,
                                         options.enc_hidden_size))

        info.write("Decoder: {} layers, {} bidirectional, {} dropout , "
                   "{} hidden \n".format(options.dec_num_layers,
                                         options.dec_bidirectional,
                                         options.dec_dropout,
                                         options.dec_hidden_size))
        info.close()

def trainer_factory(options, emb_dim, vocab_size, embeddings, pad_index,
                    sos_index, checkpoint_dir=None, device=DEVICE):

    encoder = Encoder(input_size=emb_dim, vocab_size=vocab_size,
                      hidden_size=options.enc_hidden_size,
                      embeddings=embeddings,
                      embeddings_dropout=options.embeddings_dropout,
                      finetune_embeddings=options.enc_finetune_embeddings,
                      num_layers=options.enc_num_layers,
                      bidirectional=options.enc_bidirectional,
                      dropout=options.enc_dropout,
                      rnn_type=options.enc_rnn_type,
                      device=DEVICE)
    decoder = Decoder(emb_size=emb_dim, vocab_size=vocab_size,
                      hidden_size=options.dec_hidden_size,
                      embeddings=embeddings,
                      embeddings_dropout=options.embeddings_dropout,
                      finetune_embeddings=options.dec_finetune_embeddings,
                      num_layers=options.dec_num_layers,
                      bidirectional=options.dec_bidirectional,
                      dropout=options.dec_dropout,
                      rnn_type=options.dec_rnn_type,
                      device=DEVICE)

    model = Seq2Seq(encoder, decoder, sos_index, device,
                    shared_emb=options.shared_emb)

    numparams = sum([p.numel() for p in model.parameters()])
    train_numparams = sum([p.numel() for p in model.parameters() if
                       p.requires_grad])
    print('Total Parameters: {}'.format(numparams))
    print('Trainable Parameters: {}'.format(train_numparams))
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=options.lr, weight_decay=1e-6)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_index,reduction='sum')

    trainer = Seq2SeqTrainerEpochs(model, optimizer, criterion, patience=5,
                                checkpoint_dir=checkpoint_dir,
                                decreasing_tc=options.decr_tc_ratio,
                                device=device)
    return trainer


if __name__ == '__main__':
    # --- fix argument parser default values --
    parser = argparse.ArgumentParser(description='Main options')

    # select dataset and preprocessing
    parser.add_argument('-dataset', type=str, help='Dataset used',
                        required=True)
    parser.add_argument('-preprocess', action='store_true', default=False,
                        help='Preprocess dataset used')

    # epochs to run and checkpoint to save model
    parser.add_argument('-epochs', type=int, help='epochs to train the model',
                        required=True)
    parser.add_argument('-ckpt', type=str, help='Model checkpoint',
                        required=True)
    parser.add_argument('-lr', type=float, default=0.0001, help='learning rate',
                        required=True)

    # embeddings options
    parser.add_argument('-embeddings', type=str, default=None,
                        help='Embeddings file(optional)')
    parser.add_argument('-emb_dim', type=int, help='Embeddings dimension',
                        required=True)
    parser.add_argument('-embdrop', dest='embeddings_dropout', type=float,
                        default=0, help='embeddings dropout')
    parser.add_argument('-shared_emb', action='store_true',
                        default=False, help='shared embedding layer')

    # Encoder options
    parser.add_argument('-enchidden', dest='enc_hidden_size', type=int,
                        default=256, help='encoder hidden size')
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
    parser.add_argument('-enctype', dest='enc_rnn_type',
                        default='gru', help='bidirectional enc')

    # Decoder options
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
    parser.add_argument('-dectype', dest='dec_rnn_type',
                        default='gru', help='decoder rnn type')

    # Teacher forcing options
    parser.add_argument('-tc_ratio', dest='teacherforcing_ratio',
                        default=1., type=float, help='teacher forcing ratio')
    parser.add_argument('-decr_tc_ratio', action='store_true', default=False,
                        help='decreasing teacherforcing ratio during training and val')

    options = parser.parse_args()

    # ---  read data to create vocabulary dict ---
    tokenizer = DialogSpacyTokenizer(lower=True,
                                     specials=DIALOG_SPECIAL_TOKENS)


    if options.dataset == "dailydialog":
        dataset = DailyDialogDatasetTuples('./data/ijcnlp_dailydialog',
                                           transforms=None)
    else:
        raise NameError

    dataset.normalize_data()
    if options.preprocess:
        dataset.threshold_data(30, tokenizer=tokenizer)
        dataset.trim_words(3, tokenizer=tokenizer)
    vocab_dict = dataset.create_vocab_dict(tokenizer)

    # load embeddings from file or set None (to be randomly init)
    if options.embeddings is not None:
        new_emb_file = './cache/new_embs.txt'
        old_emb_file = options.embeddings
        freq_words_file = './cache/freq_words.txt'
        emb_dim = options.emb_dim
        create_emb_file(new_emb_file, old_emb_file, freq_words_file, vocab_dict,
                        most_freq=10000)
        word2idx, idx2word, embeddings = EmbeddingsLoader(new_emb_file, emb_dim,
                                                          extra_tokens=
                                                          DIALOG_SPECIAL_TOKENS
                                                          ).load()
    else:
        word2idx, idx2word = word2idx_from_dataset(vocab_dict,
                                                   most_freq=10000,
                                                   extra_tokens=
                                                   DIALOG_SPECIAL_TOKENS)
        embeddings = None
        emb_dim = options.emb_dim
    vocab_size = len(word2idx)

    # --- set dataset transforms ---
    tokenizer = DialogSpacyTokenizer(lower=True,
                                     append_eos=True,
                                     specials=DIALOG_SPECIAL_TOKENS)
    to_token_ids = ToTokenIds(word2idx, specials=DIALOG_SPECIAL_TOKENS)
    to_tensor = ToTensor()
    dataset = dataset.map(tokenizer).map(to_token_ids).map(to_tensor)
    print("Dataset size: {}".format(len(dataset)))
    print("Vocabulary size: {}".format(vocab_size))
    import ipdb;ipdb.set_trace()

    # --- make train and val loaders ---
    collator_fn = Seq2SeqCollator(device='cpu')
    train_loader, val_loader = train_test_split(dataset,
                                                batch_train=BATCH_TRAIN_SIZE,
                                                batch_val=BATCH_VAL_SIZE,
                                                collator_fn=collator_fn,
                                                test_size=0.2)

    pad_index = word2idx[DIALOG_SPECIAL_TOKENS.PAD.value]
    sos_index = word2idx[DIALOG_SPECIAL_TOKENS.SOS.value]
    eos_index = word2idx[DIALOG_SPECIAL_TOKENS.EOS.value]
    unk_index = word2idx[DIALOG_SPECIAL_TOKENS.UNK.value]
    print("sos index {}".format(sos_index))
    print("eos index {}".format(eos_index))
    print("pad index {}".format(pad_index))
    print("unk index {}".format(unk_index))

    # --- make model and train it ---
    checkpoint_dir = options.ckpt
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print_model_info(options.ckpt, dataset, vocab_size, options)
    if options.embeddings is None:
        with open(os.path.join(checkpoint_dir, 'word2idx.pickle'), 'wb') as \
                file1:
            pickle.dump(word2idx, file1, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(checkpoint_dir, 'idx2word.pickle'), 'wb') as \
                file2:
            pickle.dump(idx2word, file2, protocol=pickle.HIGHEST_PROTOCOL)

    trainer = trainer_factory(options, emb_dim, vocab_size, embeddings,
                              pad_index, sos_index, checkpoint_dir,
                              device=DEVICE)

    trainer.fit(train_loader, val_loader, epochs=options.epochs)
    print("data stored in: {}\n".format(checkpoint_dir))