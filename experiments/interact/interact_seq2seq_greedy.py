import torch
import argparse
import unicodedata
import re
import os
import pickle

from slp.data.transforms import DialogSpacyTokenizer, ToTokenIds, ToTensor
from slp.util import from_checkpoint
from slp.util.embeddings import EmbeddingsLoader
from slp.config.special_tokens import DIALOG_SPECIAL_TOKENS
from slp.modules.seq2seq.seq2seq import Encoder,Decoder,Seq2Seq
from slp.modules.seq2seq.inference import GreedySeq2Seq

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

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


def create_model(options, embeddings, emb_dim, vocab_size, sos_index, eos_index,
                 device):

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
                      attention=options.decattn,
                      device=DEVICE)

    model = Seq2Seq(encoder, decoder, sos_index, eos_index, device,
                    shared_emb=options.shared_emb)
    return model


def create_searcher(options,model, device):
    searcher = GreedySeq2Seq(model,options.maxseqlen,device)
    return searcher


def load_embeddings(emb_file, emb_dim):
    loader = EmbeddingsLoader(emb_file, emb_dim,
                              extra_tokens=DIALOG_SPECIAL_TOKENS)
    word2idx, idx2word, embeddings = loader.load()
    return word2idx, idx2word, embeddings


def evaluate(searcher, idx2word, sentence1, device):

    indexes_batch = sentence1
    input_batch = torch.unsqueeze(indexes_batch, 0)
    lengths = torch.tensor([len(indexes) for indexes in input_batch])
    input_batch1 = input_batch.to(device)
    lengths1 = lengths.to(device)

    # Decode sentence with searcher
    tokens,logits = searcher(input_batch1, lengths1)
    decoded_words = [idx2word[token.item()] for token in tokens]

    return decoded_words


def evaluate_input(searcher, word2idx, idx2word, device):
    tokenizer = DialogSpacyTokenizer(lower=True, specials=DIALOG_SPECIAL_TOKENS)
    to_token_ids = ToTokenIds(word2idx, specials=DIALOG_SPECIAL_TOKENS)
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
    pad_index = word2idx[DIALOG_SPECIAL_TOKENS.PAD.value]
    sos_index = word2idx[DIALOG_SPECIAL_TOKENS.SOS.value]
    eos_index = word2idx[DIALOG_SPECIAL_TOKENS.EOS.value]
    print("Loaded Embeddings...")
    #  --- load model using loaded embeddings ---
    model = create_model(modeloptions, embeddings, emb_dim, vocab_size,
                         sos_index, eos_index, device)
    model = from_checkpoint(modelcheckpoint, model, map_location='cpu')
    model = model.to(device)
    print("Loaded Model...")
    # --- create searcher for encoding user's input and for providing an
    # answer ---
    searcher = create_searcher(modeloptions,model, device)
    searcher = searcher.to(device)
    searcher.eval()
    print("Interacting...")
    evaluate_input(searcher, word2idx, idx2word, device)


def make_arg_parser():
    # --- fix argument parser default values --
    parser = argparse.ArgumentParser(description='Main options')

    parser.add_argument('-modelckpt', type=str, help='Model checkpoint',
                        required=True)
    parser.add_argument('-ckpt', type=str, help='checkpoint folder',
                        required=True)
    parser.add_argument('-maxseqlen', type=int, default=15,
                        help='max seq len to be generated')


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
    parser.add_argument('-decattn', action='store_true',
                        default=False, help='decoder luong attn')
    return parser


if __name__ == '__main__':

    parser = make_arg_parser()
    options = parser.parse_args()

    input_interaction(options,
                      options.embeddings,
                      options.emb_dim,
                      options.modelckpt,
                      options.ckpt,
                      DEVICE)
