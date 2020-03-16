import os
import torch
import argparse
from torch.optim import Adam
from ignite.metrics import Loss

from slp.config.special_tokens import HRED_SPECIAL_TOKENS
from slp.data.utils import train_test_split
from slp.data.transforms import DialogSpacyTokenizer, ToTokenIds, ToTensor
from slp.data.SubtleTriples import SubTle
from slp.data.collators import HRED_Subtle_Collator
from slp.util.embeddings import EmbeddingsLoader, create_emb_file

from slp.modules.loss import SequenceCrossEntropyLoss, Perplexity
from slp.modules.seq2seq.hredseq2seq import HREDSeq2Seq, HREDSeq2Seq_Context
from slp.trainer.trainer import HREDIterationsTrainer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
MAX_EPOCHS = 20
BATCH_TRAIN_SIZE = 16
BATCH_VAL_SIZE = 16


def trainer_factory(options, emb_dim, vocab_size, embeddings, pad_index,
                    sos_index, checkpoint_dir=None, device=DEVICE):

    model = HREDSeq2Seq(options, emb_dim, vocab_size, embeddings, embeddings,
                        sos_index, device)

    numparams = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('Trainable Parameters: {}'.format(numparams))

    # print("hred model:\n{}".format(model))

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-6)

    criterion = SequenceCrossEntropyLoss(pad_index)
    perplexity = Perplexity(pad_index)
    metrics = [perplexity]

    trainer = HREDIterationsTrainer(model, optimizer, criterion, metrics,
                                    checkpoint_dir=checkpoint_dir,
                                    save_every=1000, print_every=200,
                                    device=device)
    return trainer


if __name__ == '__main__':

    # --- fix argument parser default values --
    parser = argparse.ArgumentParser(description='HRED parameter options')
    parser.add_argument('-n', dest='name', help='enter suffix for model files')

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
    parser.add_argument('-sl', dest='samplelimit', type=int,
                        default=100000, help='sample limit used for training')

    options = parser.parse_args()

    # ---  read data to create vocabulary dict ---

    tokenizer = DialogSpacyTokenizer(lower=True, prepend_sos=True,
                                     append_eos=True,
                                     specials=HRED_SPECIAL_TOKENS)

    dataset = SubTle(
        "./data/corpus0sDialogues.txt", samples_limit=options.samplelimit,
        transforms=[
            tokenizer])
    vocab_dict = dataset.create_vocab_dict(tokenizer)

    # --- create new embedding file ---

    new_emb_file = './cache/new_embs.txt'
    old_emb_file = './cache/glove.6B.50d.txt'
    freq_words_file = './cache/freq_words.txt'
    emb_dim = 50

    create_emb_file(new_emb_file, old_emb_file, freq_words_file, vocab_dict,
                    most_freq=10000)

    # --- load new embeddings! ---

    word2idx, idx2word, embeddings = EmbeddingsLoader(new_emb_file, emb_dim,
                                                      extra_tokens=
                                                      HRED_SPECIAL_TOKENS
                                                      ).load()
    vocab_size = len(word2idx)
    print("Vocabulary size: {}".format(vocab_size))

    # --- read dataset again and apply transforms ---

    to_token_ids = ToTokenIds(word2idx, specials=HRED_SPECIAL_TOKENS)
    to_tensor = ToTensor()
    dataset = SubTle("./data/corpus0sDialogues.txt",
                     samples_limit=options.samplelimit,
                     transforms=[tokenizer, to_token_ids, to_tensor])

    print("Dataset size: {}".format(len(dataset)))
    # --- make train and val loaders ---

    collator_fn = HRED_Subtle_Collator(device='cpu')
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
    if options.name is None:
        assert False, "Give model name for checkpoint!"

    checkpoint_dir = os.path.join('./checkpoints/hred/pretrained', options.name)
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

    trainer = trainer_factory(options, emb_dim, vocab_size, embeddings,
                              pad_index, sos_index, checkpoint_dir,
                              device=DEVICE)

    trainer.fit(train_loader, val_loader, n_iters=8000)
