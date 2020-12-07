import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pickle
import sys
from torch.utils.data import DataLoader
from slp.data.vocab import Vocab, word2idx_from_dataset
from slp.util.embeddings import EmbeddingsLoader, create_emb_file
from slp.config.special_tokens import DIALOG_SPECIAL_TOKENS
from slp.data.transforms import DialogSpacyTokenizer, ToTokenIds, ToTensor
from slp.data.DailyDialog import SubsetDailyDialogDatasetEmoTuples
from slp.data.moviecorpus import SubsetMovieCorpusTuples
from slp.data.collators import NoEmoSeq2SeqCollator, Seq2SeqCollator
from torch.optim import Adam
from slp.modules.loss import SequenceCrossEntropyLoss, Perplexity
from slp.trainer.seq2seqtrainer import Seq2SeqIterationsTrainer,\
    Seq2SeqTrainerEpochs
from slp.modules.seq2seq.seq2seq import Encoder, Decoder, Seq2Seq
from slp.modules.metrics import BleuMetric, DistinctN

"""
This script is used to train your seq2seq model using ready data pickles!!
"""

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
BATCH_TRAIN_SIZE = 64
BATCH_VAL_SIZE = 64


def load_datasets_from_pickle(options):
    if os.path.exists(options.datasetfolder):
        if options.datasetname == 'dailydialog':
            with open(os.path.join(options.datasetfolder, 'train_set.pkl'),
                      'rb')as \
                    handle:
                train_list = pickle.load(handle)
                train_dataset = SubsetDailyDialogDatasetEmoTuples(train_list)
            handle.close()
            with open(os.path.join(options.datasetfolder, 'val_set.pkl'),
                      'rb')as \
                    handle:
                val_list = pickle.load(handle)
                val_dataset = SubsetDailyDialogDatasetEmoTuples(val_list)

            handle.close()
        elif options.datasetname == 'moviecorpus':
            with open(os.path.join(options.datasetfolder, 'train_set.pkl'),
                      'rb')as \
                    handle:
                train_list = pickle.load(handle)
                train_dataset = SubsetMovieCorpusTuples(train_list)
            handle.close()
            with open(os.path.join(options.datasetfolder, 'val_set.pkl'),
                      'rb')as \
                    handle:
                val_list = pickle.load(handle)
                val_dataset = SubsetMovieCorpusTuples(val_list)

            handle.close()
        else:
            print("Given datasetname is not implemented!")
            raise NotImplementedError
    else:
        raise FileNotFoundError
    return train_dataset,val_dataset


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


def trainer_factory(options, emb_dim, vocab_size, embeddings, vocab,
                    checkpoint_dir=None, device=DEVICE):

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
                      start_idx=vocab.start_idx,
                      embeddings=embeddings,
                      embeddings_dropout=options.embeddings_dropout,
                      finetune_embeddings=options.dec_finetune_embeddings,
                      num_layers=options.dec_num_layers,
                      bidirectional=options.dec_bidirectional,
                      dropout=options.dec_dropout,
                      rnn_type=options.dec_rnn_type,
                      attention=options.decattn,
                      device=DEVICE)

    model = Seq2Seq(encoder, decoder, vocab.start_idx, vocab.end_idx, device,
                    shared_emb=options.shared_emb)
    torch.save(model.state_dict(), os.path.join(
        options.ckpt, '{}_{}.pth'.format(0, 'model_checkpoint')))

    numparams = sum([p.numel() for p in model.parameters()])
    train_numparams = sum([p.numel() for p in model.parameters() if
                       p.requires_grad])
    print('Total Parameters: {}'.format(numparams))
    print('Trainable Parameters: {}'.format(train_numparams))
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=options.lr, weight_decay=1e-5)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=4, gamma=0.5)
    criterion = SequenceCrossEntropyLoss(vocab.pad_idx)

    metrics = {'bleu':BleuMetric(), 'distinct':DistinctN()}
    mytrainer = Seq2SeqTrainerEpochs(model,
                                     optimizer,
                                     criterion,
                                     scheduler=exp_lr_scheduler,
                                     patience=10,
                                     checkpoint_dir=checkpoint_dir,
                                     clip=3,
                                     metrics=metrics,
                                     vocab=vocab,
                                     best_metric='bleu',
                                     device=device,
                                     genoptions=options)

    # bleum = BLEU_metric_generation(BATCH_VAL_SIZE,pad_index,idx2word)
    # trainer = Seq2SeqIterationsTrainer(model, optimizer, criterion,
    #                                    perplexity=True, clip=5,
    #                                    checkpoint_dir=checkpoint_dir,
    #                                    device=device,bleu=bleum)

    return mytrainer


def make_arg_parser():
    # --- fix argument parser default values --
    parser = argparse.ArgumentParser(description='Main options')

    # dataset file (pickles)
    modeloptions = parser.add_argument_group('Train and model options')
    modeloptions.add_argument('-datasetfolder', type=str, help='Dataset folder ('
                                                        'pickles) used',
                        required=True)
    modeloptions.add_argument('-datasetname', type=str, help='Dataset name',
                        required=True)

    # epochs to run and checkpoint to save model
    modeloptions.add_argument('-iters', type=int, help='iters to train the model',
                        required=True)
    modeloptions.add_argument('-ckpt', type=str, help='Model checkpoint',
                        required=True)
    modeloptions.add_argument('-lr', type=float, default=0.0001, help='learning rate',
                        required=True)

    # embeddings options
    modeloptions.add_argument('-embeddings', type=str, default=None,
                        help='Embeddings file(optional)')
    modeloptions.add_argument('-emb_dim', type=int, help='Embeddings dimension',
                        required=True)
    modeloptions.add_argument('-embdrop', dest='embeddings_dropout', type=float,
                        default=0, help='embeddings dropout')
    modeloptions.add_argument('-shared_emb', action='store_true',
                        default=False, help='shared embedding layer')

    # Encoder options
    modeloptions.add_argument('-enchidden', dest='enc_hidden_size', type=int,
                        default=256, help='encoder hidden size')
    modeloptions.add_argument('-encembtrain', dest='enc_finetune_embeddings',
                        action='store_true', default=False,
                        help='encoder finetune embeddings')
    modeloptions.add_argument('-encnumlayers', dest='enc_num_layers', type=int,
                        default=1, help='encoder number of layers')
    modeloptions.add_argument('-encbi', dest='enc_bidirectional',
                        action='store_true',
                        default=False, help='bidirectional enc')
    modeloptions.add_argument('-encdrop', dest='enc_dropout', type=float,
                        default=0, help='encoder dropout')
    modeloptions.add_argument('-enctype', dest='enc_rnn_type',
                        default='gru', help='bidirectional enc')

    # Decoder options
    modeloptions.add_argument('-dechidden', dest='dec_hidden_size',
                        type=int,
                        default=256, help='decoder hidden size')
    modeloptions.add_argument('-decembtrain', dest='dec_finetune_embeddings',
                        action='store_true',
                        default=False, help='decoder finetune embeddings')
    modeloptions.add_argument('-decnumlayers', dest='dec_num_layers',
                        type=int,
                        default=1, help='decoder number of layers')
    modeloptions.add_argument('-decbi', dest='dec_bidirectional',
                        action='store_true',
                        default=False, help='bidirectional decoder')
    modeloptions.add_argument('-decdrop', dest='dec_dropout', type=float,
                        default=0, help='decoder dropout')
    modeloptions.add_argument('-dectype', dest='dec_rnn_type',
                        default='gru', help='decoder rnn type')
    modeloptions.add_argument('-decattn', action='store_true',
                        default=False, help='decoder luong attn')

    # Teacher forcing options
    modeloptions.add_argument('-tc_ratio', dest='teacherforcing_ratio',
                        default=1, type=float, help='teacher forcing ratio')
    modeloptions.add_argument('-decr_tc_ratio', action='store_true', default=False,
                        help='decreasing teacherforcing ratio during training and val')

    gener = parser.add_argument_group('Generation options')
    gener.add_argument(
        '-beam_size',
        type=int,
        default=1,
        help='Beam size, if 1 then greedy search'
    )
    gener.add_argument(
        '-beam_min_length',
        type=int,
        default=1,
        help='Minimum length of prediction to be generated by the beam search'
    )
    gener.add_argument(
        '-beam_context_block_ngram',
        type=int,
        default=-1,
        help=(
            'Size n-grams to block in beam search from the context. val <= 0 '
            'implies no blocking **NOT USED**'
        )
    )
    gener.add_argument(
        '-beam_block_ngram',
        type=int,
        default=-1,
        help='Size n-grams to block in beam search. val <= 0 implies no '
             'blocking **NOT USED**'
    )
    gener.add_argument(
        '-beam_length_penalty',
        type=float,
        default=0.75,
        help='Applies a length penalty. Set to 0 for no penalty.'
    )
    gener.add_argument(
        '-skip_generation',
        default=False,
        help='Skip beam search. Useful for speeding up training, '
             'if perplexity is the validation metric.'
    )
    gener.add_argument(
        '-method',
        choices={'beam', 'greedy', 'topk', 'nucleus', 'delayedbeam'},
        default='greedy',
        help='Generation algorithm'
    )
    gener.add_argument(
        '-topk', type=int, default=10, help='K used in Top K sampling'
    )
    gener.add_argument(
        '-topp', type=float, default=0.9, help='p used in nucleus sampling'
    )
    gener.add_argument(
        '-beam_delay', type=int, default=30, help='used in delayedbeam search'
    )
    gener.add_argument(
        '-temperature',
        type=float,
        default=1.0,
        help='temperature to add during decoding'
    )
    gener.add_argument(
        '-compute_tokenized_bleu',
        default=False,
        help='if true, compute tokenized bleu scores **NOT USED**'
    )
    gener.add_argument(
        '-maxlen',
        type=int,
        default=20,
        help='max length of sequence to be generated'
    )
    gener.add_argument(
        '-N_best',
        type=int,
        default=1,
        help='N best answers to take after beamsearch'
    )
    return parser

if __name__ == '__main__':

    parser = make_arg_parser()
    options = parser.parse_args()

    # ---  read data to create vocabulary dict ---
    tokenizer = DialogSpacyTokenizer(lower=True,
                                     specials=DIALOG_SPECIAL_TOKENS)
    train_dataset, val_dataset = load_datasets_from_pickle(options)
    vocab_dict = train_dataset.create_vocab_dict(tokenizer)

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
    vocab = Vocab(word2idx, idx2word, DIALOG_SPECIAL_TOKENS)
    vocab_size = len(vocab)

    # --- set dataset transforms ---
    tokenizer = DialogSpacyTokenizer(lower=True,
                                     append_eos=True,
                                     specials=DIALOG_SPECIAL_TOKENS)
    to_token_ids = ToTokenIds(word2idx, specials=DIALOG_SPECIAL_TOKENS)
    to_tensor = ToTensor()
    train_dataset = train_dataset.map(tokenizer).map(to_token_ids).map(
        to_tensor)
    val_dataset = val_dataset.map(tokenizer).map(to_token_ids).map(
        to_tensor)
    print("Dataset size: {}".format(len(train_dataset)))
    print("Vocabulary size: {}".format(vocab_size))

    # --- make train and val loaders ---
    if options.datasetname =='dailydialog':
        collator_fn = NoEmoSeq2SeqCollator(device='cpu')
    elif options.datasetname == 'moviecorpus':
        collator_fn = Seq2SeqCollator(device='cpu')
    else:
        print("Given datasetname is not implemented!")
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, batch_size=BATCH_TRAIN_SIZE,
                              collate_fn=collator_fn,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_TRAIN_SIZE,
                            collate_fn=collator_fn,drop_last=True)

    print("sos index {}".format(vocab.start_idx))
    print("eos index {}".format(vocab.end_idx))
    print("pad index {}".format(vocab.pad_idx))
    print("unk index {}".format(vocab.unk_idx))

    # --- make model and train it ---
    checkpoint_dir = options.ckpt
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print_model_info(options.ckpt, train_dataset, vocab_size, options)
    if options.embeddings is None:
        with open(os.path.join(checkpoint_dir, 'word2idx.pickle'), 'wb') as \
                file1:
            pickle.dump(word2idx, file1, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(checkpoint_dir, 'idx2word.pickle'), 'wb') as \
                file2:
            pickle.dump(idx2word, file2, protocol=pickle.HIGHEST_PROTOCOL)

    trainer = trainer_factory(options, emb_dim, vocab_size, embeddings,
                              vocab, checkpoint_dir,
                              device=DEVICE)

    trainer.fit(train_loader, val_loader, epochs=options.iters)

    for epoch in range(options.iters):
        print(trainer.metrics_recoder[epoch])
    print("data stored in: {}\n".format(checkpoint_dir))