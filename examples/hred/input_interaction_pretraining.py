import torch
import argparse
from slp.data.transforms import DialogSpacyTokenizer, ToTokenIds, ToTensor

from slp.util import from_checkpoint
from slp.util.embeddings import EmbeddingsLoader
from slp.config.special_tokens import HRED_SPECIAL_TOKENS
from slp.modules.seq2seq.hredseq2seq import HREDSeq2Seq, GreedySearchHREDSeq2Seq

import unicodedata
import re
import os
import pickle

def unicodeToAscii( s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString( s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def create_model(modeloptions, embeddings, emb_dim, vocab_size, sos_index,
                 device):

    model = HREDSeq2Seq(modeloptions, emb_dim, vocab_size, embeddings,
                 embeddings, sos_index, device)
    return model

from slp.modules.seq2seq.greedy import GreedySearchSeq2Seq
def create_searcher(model, device):
    searcher = GreedySearchSeq2Seq(model, device)
    return searcher


def load_embeddings(emb_file, emb_dim):
    loader = EmbeddingsLoader(emb_file, emb_dim, extra_tokens=HRED_SPECIAL_TOKENS)
    word2idx, idx2word, embeddings = loader.load()
    return word2idx, idx2word, embeddings


def evaluate(searcher, idx2word, sentence1, device):

    indexes_batch = sentence1
    input_batch = torch.unsqueeze(indexes_batch, 0)
    lengths = torch.tensor([len(indexes) for indexes in input_batch])
    input_batch1 = input_batch.to(device)
    lengths1 = lengths.to(device)

    empty_input = torch.zeros(input_batch1.shape[0], input_batch1.shape[1],
                              device=device).long()
    len_empty = torch.tensor([len(s) for s in empty_input],
                             device=device)
    # Decode sentence with searcher
    tokens, scores = searcher(empty_input, len_empty, input_batch1, lengths1)
    decoded_words = [idx2word[token.item()] for token in tokens]

    return decoded_words


def evaluate_input(searcher, word2idx, idx2word, device):
    tokenizer = DialogSpacyTokenizer(lower=True, specials=HRED_SPECIAL_TOKENS)
    to_token_ids = ToTokenIds(word2idx, specials=HRED_SPECIAL_TOKENS)
    to_tensor = ToTensor()
    transforms = [tokenizer, to_token_ids, to_tensor]

    while True:
        try:
            # Get input sentence
            input_sentence1 = input('> ')
            if input_sentence1 == 'q' or input_sentence1 == 'quit': break

            # Normalize sentence
            input_sentence1 = normalizeString(input_sentence1)

            # Evaluate sentence
            for t in transforms:
                input_sentence1 = t(input_sentence1)

            output_words = evaluate(searcher, idx2word, input_sentence1,
                                    device)

            print(output_words)
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


def input_interaction(modeloptions, embfile, emb_dim, modelcheckpoint,
                      checkpointfolder, device):

    if embfile is not None:
        print("Embedding file given! Load embeddings...")
        word2idx, idx2word, embeddings = load_embeddings(embfile, emb_dim)
    else:
        print("Embedding file not given! Load embedding dicts for checkpoint "
              "folder...")
        embeddings = None
        with open(os.path.join(checkpointfolder, 'word2idx.pickle'), 'rb') as \
                handle:
            word2idx = pickle.load(handle)
        with open(os.path.join(checkpointfolder, 'idx2word.pickle'), 'rb') as \
                handle:
            idx2word = pickle.load(handle)

    vocab_size = len(word2idx)
    pad_index = word2idx[HRED_SPECIAL_TOKENS.PAD.value]
    sos_index = word2idx[HRED_SPECIAL_TOKENS.SOU.value]
    eos_index = word2idx[HRED_SPECIAL_TOKENS.EOU.value]
    print("Loaded Embeddings...")
    #  --- load model using loaded embeddings ---
    model = create_model(modeloptions, embeddings, emb_dim, vocab_size,
                         sos_index, device)
    model = from_checkpoint(modelcheckpoint, model, map_location='cpu')
    model = model.to(device)
    print("Loaded Model...")
    # --- create searcher for encoding user's input and for providing an
    # answer ---
    searcher = create_searcher(model, device)
    searcher = searcher.to(device)
    searcher.eval()
    print("Interacting...")
    evaluate_input(searcher, word2idx, idx2word, device)


if __name__ == '__main__':

    # --- fix argument parser default values --
    parser = argparse.ArgumentParser(description='HRED parameter options and'
                                                 'checkpoints')

    parser.add_argument('-modelckpt', type=str, help='Model checkpoint',
                        required=True)
    parser.add_argument('-ckpt', type=str, help='checkpoint folder',
                        required=True)

    parser.add_argument('-embeddings', type=str, default=None,
                        help='Embeddings file(optional)')
    parser.add_argument('-emb_dim', type=int, help='Embeddings dimension',
                        required=True)

    parser.add_argument('-device', type=str, help='Device cpu|cuda:X')


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
    parser.add_argument('-conthidden', dest='contenc_hidden_size',
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

    parser.add_argument('-shared', action='store_true',
                        default=False, help='shared weights between encoder '
                                            'and decoder')
    parser.add_argument('-shared_emb', action='store_true',
                        default=False, help='shared embedding layer')

    options = parser.parse_args()

    if options.pretraining is False:
        assert False, "you are using this script to test encoder-decoder " \
                      "model questions-answers."
    input_interaction(options,
                      options.embeddings,
                      options.emb_dim,
                      options.modelckpt,
                      options.ckpt,
                      options.device)


