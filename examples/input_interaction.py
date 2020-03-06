import torch
import argparse
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor

from slp.util import from_checkpoint
from slp.util.embeddings import EmbeddingsLoader
from slp.config.special_tokens import HRED_SPECIAL_TOKENS
from slp.modules.seq2seq.hred import HRED


def create_model(modeloptions, embeddings, emb_dim, vocab_size, sos_index,
                 device):

    model = HRED(modeloptions, emb_dim, vocab_size, embeddings,
                 embeddings, sos_index, device)
    return model


def create_searcher(model):
    searcher = model
    return searcher


def load_embeddings(emb_file, emb_dim):
    loader = EmbeddingsLoader(emb_file, emb_dim, extra_tokens=HRED_SPECIAL_TOKENS)
    word2idx, idx2word, embeddings = loader.load()
    return word2idx, idx2word, embeddings


# def load_model_from_checkpoint(embfile, checkpointfile,device):
#
#     word2idx, idx2word, embeddings = load_embeddings(embfile)
#
#     model = create_model(embeddings)
#     model = from_checkpoint(checkpointfile, model, map_location='cpu')
#     model = model.to(device)
#
#     return model

def evaluate(searcher, idx2word, sentence, history, device):
    max_length = 11
    # Format input sentence as a batch
    indexes_batch = sentence
    input_batch = torch.unsqueeze(indexes_batch, 0)

    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in input_batch])


    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    import ipdb;ipdb.set_trace()

    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [idx2word[token.item()] for token in tokens]

    return decoded_words


def evaluate_input(searcher, word2idx, idx2word, device):
    tokenizer = SpacyTokenizer(specials=HRED_SPECIAL_TOKENS)
    to_token_ids = ToTokenIds(word2idx, specials=HRED_SPECIAL_TOKENS)
    to_tensor = ToTensor()
    transforms = [tokenizer, to_token_ids, to_tensor]
    history = []

    while True:
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            #input_sentence = normalizeString(input_sentence)

            # Evaluate sentence
            for t in transforms:
                input_sentence = t(input_sentence)

            history.append(input_sentence)

            output_words = evaluate(searcher, idx2word, input_sentence,
                                    history, device)

            # Format and print response sentence
            print(output_words)
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


def input_interaction(modeloptions, embfile, emb_dim, checkpointfile,
                      outputfile, device):

    word2idx, idx2word, embeddings = load_embeddings(embfile, emb_dim)
    vocab_size = len(word2idx)
    pad_index = word2idx[HRED_SPECIAL_TOKENS.PAD.value]
    sos_index = word2idx[HRED_SPECIAL_TOKENS.SOU.value]
    eos_index = word2idx[HRED_SPECIAL_TOKENS.EOU.value]
    print("Loaded Embeddings...")
    #  --- load model using loaded embeddings ---
    model = create_model(modeloptions, embeddings, emb_dim, vocab_size,
                         sos_index, device)
    model = from_checkpoint(checkpointfile, model, map_location='cpu')
    model = model.to(device)
    print("Loaded Model...")
    # --- create searcher for encoding user's input and for providing an
    # answer ---
    searcher = create_searcher(model)
    searcher.eval()
    print("Interacting...")
    evaluate_input(searcher, word2idx, idx2word, device)


if __name__ == '__main__':

    # --- fix argument parser default values --
    parser = argparse.ArgumentParser(description='HRED parameter options and'
                                                 'checkpoints')

    parser.add_argument('--ckpt', type=str, help='Model checkpoint')
    parser.add_argument('--embeddings', type=str, help='Embeddings file')
    parser.add_argument('--emb_dim', type=int, help='Embeddings dimension')

    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--device', type=str, help='Device cpu|cuda:X')

    parser.add_argument('-n', dest='name', default='hred', help='enter '
                                                                 'suffix for ' \
                                                      'model files')
    parser.add_argument('-model_path', dest='model_path', default='./models', help='enter the path in which you want to store the model state')
    parser.add_argument('-enchidden', dest='enc_hidden_size',
                        action='store_true',
                        default=256, help='encoder hidden size')
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
                        default=256, help='context encoder input size')
    parser.add_argument('-conthiddensize', dest='contenc_hidden_size',
                        action='store_true',
                        default=256, help='context encoder hidden size')
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
                        default=256, help='decoder hidden size')
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

    input_interaction(options,
                      options.embeddings,
                      options.emb_dim,
                      options.ckpt,
                      options.output,
                      options.device)
