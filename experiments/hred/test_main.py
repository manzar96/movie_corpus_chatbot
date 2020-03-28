import os
import torch
import torch.nn as nn
import argparse
from torch.optim import Adam
from ignite.metrics import Loss
import pickle
from slp.config.special_tokens import HRED_SPECIAL_TOKENS
from slp.data.utils import train_test_split
from slp.data.transforms import DialogSpacyTokenizer, ToTokenIds, ToTensor
from slp.data.collators import HRED_Collator
from slp.util.embeddings import EmbeddingsLoader, create_emb_file
from slp.data.vocab import word2idx_from_dataset

from slp.modules.loss import SequenceCrossEntropyLoss, Perplexity
from slp.modules.seq2seq.test_seq2seq import Seq2Seq
from slp.trainer.trainer import HREDTrainer,HREDTrainerEpochsTest

from slp.data.Semaine import SemaineDatasetTriplesOnly
from slp.data.moviecorpus import MovieCorpusDatasetTriples
from slp.data.DailyDialog import DailyDialogDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
BATCH_TRAIN_SIZE = 64
BATCH_VAL_SIZE = 34

def init_param(model):
    for name, param in model.named_parameters():
        # skip over the embeddings so that the padding index ones are 0
        if 'embed' in name:
            continue
        elif ('rnn' in name or 'lm' in name) and len(param.size()) >= 2:
            nn.init.orthogonal_(param)
        else:
            nn.init.normal_(param, 0, 0.01)

def trainer_factory(options, emb_dim, vocab_size, pad_index,
                    sos_index,embeddings=None, checkpoint_dir=None,
                    device=DEVICE):

    model = Seq2Seq(options)

    numparams = sum([p.numel() for p in model.parameters() if p.requires_grad])
    init_param(model)
    print('Trainable Parameters: {}'.format(numparams))
    import ipdb;ipdb.set_trace()
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=1e-6)

    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    perplexity = Perplexity(pad_index)

    metrics = {
        'loss': Loss(criterion),
        'ppl': Loss(perplexity)}

    # trainer = HREDTrainer(model, optimizer,
    #                       checkpoint_dir=checkpoint_dir, metrics=metrics,
    #                       non_blocking=True, retain_graph=False,
    #                       patience=5,
    #                       device=device, loss_fn=criterion)

    trainer = HREDTrainerEpochsTest(model, optimizer, criterion,patience=5,
                                    checkpoint_dir=checkpoint_dir,
                                    device=device)

    return trainer


if __name__ == '__main__':
    # --- fix argument parser default values --
    parser = argparse.ArgumentParser(description='Main options')
    parser.add_argument('-dataset', type=str, help='Dataset used',
                        required=True)
    parser.add_argument('-preprocess', action='store_true', default=False,
                        help='Preprocess dataset used')
    parser.add_argument('-ckpt', type=str, help='Model checkpoint',
                        required=True)
    parser.add_argument('-embeddings', type=str, default=None,
                        help='Embeddings file(optional)')
    parser.add_argument('-emb_dim', type=int, help='Embeddings dimension',
                        required=True)
    parser.add_argument('-epochs', type=int, help='iterations for training'
                                                  'in mini batches',
                        required=True)

    parser.add_argument('-n', dest='name', help='enter suffix for model files', required=True)
    parser.add_argument('-e', dest='epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('-pt', dest='patience', type=int, default=-1, help='validtion patience for early stopping default none')
    parser.add_argument('-tc', dest='teacher', action='store_true', default=False, help='default teacher forcing')
    parser.add_argument('-bi', dest='bidi', action='store_true', default=False, help='bidirectional enc/decs')
    parser.add_argument('-test', dest='test', action='store_true', default=False, help='only test or inference')
    parser.add_argument('-shrd_dec_emb', dest='shrd_dec_emb', action='store_true', default=False, help='shared embedding in/out for decoder')
    parser.add_argument('-btstrp', dest='btstrp', default=None, help='bootstrap/load parameters give name')
    parser.add_argument('-lm', dest='lm', action='store_true', default=False, help='enable a RNN language model joint training as well')
    parser.add_argument('-toy', dest='toy', action='store_true', default=False, help='loads only 1000 training and 100 valid for testing')
    parser.add_argument('-pretty', dest='pretty', action='store_true', default=False, help='pretty print inference')
    parser.add_argument('-mmi', dest='mmi', action='store_true', default=False, help='Using the mmi anti-lm for ranking beam')
    parser.add_argument('-drp', dest='drp', type=float, default=0.3, help='dropout probability used all throughout')
    parser.add_argument('-nl', dest='num_lyr', type=int, default=1, help='number of enc/dec layers(same for both)')
    parser.add_argument('-lr', dest='lr', type=float, default=0.001, help='learning rate for optimizer')
    parser.add_argument('-bs', dest='bt_siz', type=int, default=100, help='batch size')
    parser.add_argument('-bms', dest='beam', type=int, default=1, help='beam size for decoding')
    parser.add_argument('-vsz', dest='vocab_size', type=int, default=10004, help='size of vocabulary')
    parser.add_argument('-esz', dest='emb_size', type=int, default=300, help='embedding size enc/dec same')
    parser.add_argument('-uthid', dest='ut_hid_size', type=int, default=600, help='encoder utterance hidden state')
    parser.add_argument('-seshid', dest='ses_hid_size', type=int, default=1200, help='encoder session hidden state')
    parser.add_argument('-dechid', dest='dec_hid_size', type=int, default=600, help='decoder hidden state')



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
        dataset.threshold_data(20, tokenizer=tokenizer)
        dataset.trim_words(2, tokenizer=tokenizer)
    vocab_dict = dataset.create_vocab_dict(tokenizer)

    if options.embeddings is not None:
        import ipdb;ipdb.set_trace()
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
    else:
        word2idx, idx2word = word2idx_from_dataset(vocab_dict,
                                                   most_freq=10000,
                                                   extra_tokens=
                                                   HRED_SPECIAL_TOKENS)
        embeddings = None
        emb_dim = options.emb_dim

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
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if options.embeddings is None:
        with open(os.path.join(checkpoint_dir, 'word2idx.pickle'), 'wb') as \
                file1:
            pickle.dump(word2idx, file1, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(checkpoint_dir, 'idx2word.pickle'), 'wb') as \
                file2:
            pickle.dump(word2idx, file2, protocol=pickle.HIGHEST_PROTOCOL)

    trainer = trainer_factory(options, emb_dim, vocab_size, embeddings,
                              pad_index, sos_index, checkpoint_dir,
                              device=DEVICE)

    trainer.fit(train_loader, val_loader, epochs=options.epochs)
    print("data stored in: {}\n".format(checkpoint_dir))
