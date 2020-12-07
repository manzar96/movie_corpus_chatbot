import torch
import argparse
import unicodedata
import re
import os
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from slp.data.transforms import DialogSpacyTokenizer, ToTokenIds, ToTensor
from slp.data.vocab import Vocab,tensor2text
from slp.util import from_checkpoint
from slp.util.embeddings import EmbeddingsLoader
from slp.data.DailyDialog import SubsetDailyDialogDatasetEmoTuples
from slp.data.moviecorpus import SubsetMovieCorpusTuples
from slp.data.collators import NoEmoSeq2SeqCollator,Seq2SeqCollator
from slp.config.special_tokens import DIALOG_SPECIAL_TOKENS
from slp.modules.seq2seq.seq2seq import Encoder,Decoder,Seq2Seq
from slp.modules.metrics import BleuMetric, DistinctN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
BATCHSIZE=32

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

    model = Seq2Seq(encoder, decoder, sos_index, eos_index,
                    shared_emb=options.shared_emb, device=device)
    return model


def load_embeddings(emb_file, emb_dim):
    loader = EmbeddingsLoader(emb_file, emb_dim,
                              extra_tokens=DIALOG_SPECIAL_TOKENS)
    word2idx, idx2word, embeddings = loader.load()
    return word2idx, idx2word, embeddings


def load_dataset_from_pickle(options):
    if os.path.exists(options.datasetfolder):
        if options.datasetname == 'dailydialog':
            with open(os.path.join(options.datasetfolder, 'test_set.pkl'),
                      'rb')as \
                    handle:
                test_list = pickle.load(handle)
                test_dataset = SubsetDailyDialogDatasetEmoTuples(test_list)
            handle.close()

        elif options.datasetname == 'moviecorpus':
            with open(os.path.join(options.datasetfolder, 'test_set.pkl'),
                      'rb')as \
                    handle:
                test_list = pickle.load(handle)
                test_dataset = SubsetMovieCorpusTuples(test_list)
            handle.close()

        else:
            print("Given datasetname is not implemented!")
            raise NotImplementedError
    else:
        raise FileNotFoundError

    return test_dataset


def print_answers_to_file(outfile,gen_sent,input_sent,target_sent):
    outfile.write("Input sentence: {}\n".format(input_sent))
    outfile.write("Target sentence: {}\n".format(target_sent))
    for i in range(len(gen_sent)):
        outfile.write("Response sentence {}: {}\n".format(i,gen_sent[i]))
    outfile.write("="*10+"\n")


def test(options, test_loader, model, vocab, device):

    outputfolder = options.output
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    outfile = open(os.path.join(outputfolder, "test_results.txt"), "w")
    metrics = {'bleu':BleuMetric(), 'distinct':DistinctN()}
    with torch.no_grad():
        all_outs = []
        for index, batch in enumerate(tqdm(test_loader)):
            inputs, inputs_lengths, targets, targets_lengths = batch
            inputs = inputs.to(device)
            inputs_lengths = inputs_lengths.to(device)
            beam_preds_scores, _ = model.generate(inputs, inputs_lengths,
                                                  options,
                                                  vocab.start_idx,
                                                  vocab.end_idx,
                                                  vocab.pad_idx)
            preds, scores = zip(*beam_preds_scores)
            inp_texts = [tensor2text(inp, vocab) for inp in inputs]
            hyp_texts = [tensor2text(pred, vocab) for pred in preds]
            ref_texts = [tensor2text(target, vocab) for target in targets]
            # write data in pickle for eval scripts!
            for data in zip(inp_texts, hyp_texts, ref_texts):
                all_outs.append(data)


            bleu1 = 0
            bleu2 = 0
            bleu3 = 0
            bleu4 = 0
            distinct1 = 0
            distinct2 = 0
            distinct3 = 0
            for inp,text, target in zip(inp_texts, hyp_texts, ref_texts):
                outfile.write("Input: {}\n".format(inp))
                outfile.write("Answer: {}\n".format(text))
                outfile.write("Ref: {}\n".format(target))
                outfile.write("\n")

                if 'bleu' in metrics.keys():
                    bleu1 += (metrics['bleu'].compute([text], target, k=1) / len(
                        ref_texts))
                    bleu2 += metrics['bleu'].compute([text], target, k=2)/len(ref_texts)
                    bleu3 += metrics['bleu'].compute([text], target, k=3)/len(ref_texts)
                    bleu4 += metrics['bleu'].compute([text], target, k=4)/len(ref_texts)
                if 'distinct' in metrics.keys():
                    distinct1 += metrics['distinct'].distinct_n_sentence_level(
                        text, 1)/len(ref_texts)
                    distinct2 += metrics['distinct'].distinct_n_sentence_level(
                        text, 2)/len(ref_texts)
                    distinct3 += metrics['distinct'].distinct_n_sentence_level(
                        text, 3)/len(ref_texts)
        outfile.close()
        with open(os.path.join(outputfolder, "test_results.pkl"), "wb") as out:
            pickle.dump(all_outs, out, protocol=pickle.HIGHEST_PROTOCOL)
        bleu1 /= len(test_loader)
        bleu2 /= len(test_loader)
        bleu3 /= len(test_loader)
        bleu4 /= len(test_loader)
        distinct1 /= len(test_loader)
        distinct2 /= len(test_loader)
        distinct3 /= len(test_loader)

        test_metrics = {'bleu1': bleu1, 'bleu2': bleu2, 'bleu3': bleu3,
                        'bleu4': bleu4, 'distinct1': distinct1,
                        'distinct2': distinct2, 'distinct3': distinct3}
        return test_metrics


def make_arg_parser():
    # --- fix argument parser default values --
    parser = argparse.ArgumentParser(description='Main options')

    modeloptions = parser.add_argument_group('Data and model options')
    modeloptions.add_argument('-modelckpt', type=str, help='Model checkpoint',
                        required=True)
    modeloptions.add_argument('-ckpt', type=str, help='checkpoint folder',
                        required=True)
    modeloptions.add_argument('-output', type=str, help='output folder',
                        required=True)
    modeloptions.add_argument('-datasetfolder', type=str,required=True,
                        help="Dataset folder path where pickle is saved")
    modeloptions.add_argument('-datasetname', type=str, help='Dataset name',
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

    # --- load embeddings ---
    if options.embeddings is not None:
        print("Embedding file given! Load embeddings...")
        word2idx, idx2word, embeddings = load_embeddings(options.embeddings,
                                                         options.emb_dim)
    else:
        print("Embedding file not given! Load embedding dicts for checkpoint "
              "folder...")
        embeddings = None
        with open(os.path.join(options.ckpt, 'word2idx.pickle'), 'rb') as \
                handle:
            word2idx = pickle.load(handle)
        with open(os.path.join(options.ckpt, 'idx2word.pickle'), 'rb') as \
                handle:
            idx2word = pickle.load(handle)

    vocab = Vocab(word2idx, idx2word, DIALOG_SPECIAL_TOKENS)
    vocab_size = len(word2idx)
    pad_index = vocab.pad_idx
    sos_index = vocab.start_idx
    eos_index = vocab.end_idx
    print("sos index {}".format(vocab.start_idx))
    print("eos index {}".format(vocab.end_idx))
    print("pad index {}".format(vocab.pad_idx))
    print("unk index {}".format(vocab.unk_idx))
    print("Loaded Embeddings...")

    #  --- load model using loaded embeddings ---
    model = create_model(options, embeddings, options.emb_dim,
                         vocab_size, sos_index,eos_index, DEVICE)
    model = from_checkpoint(options.modelckpt, model, map_location='cpu')
    model = model.to(DEVICE)
    print("Loaded Model...")

    # load test dataset
    test_dataset = load_dataset_from_pickle(options)

    tokenizer = DialogSpacyTokenizer(lower=True,
                                     append_eos=True,
                                     specials=DIALOG_SPECIAL_TOKENS)
    to_token_ids = ToTokenIds(word2idx, specials=DIALOG_SPECIAL_TOKENS)
    to_tensor = ToTensor()
    test_dataset = test_dataset.map(tokenizer).map(to_token_ids).map(to_tensor)

    # --- load collator ---
    if options.datasetname =='dailydialog':
        collator_fn = NoEmoSeq2SeqCollator(device='cpu')
    elif options.datasetname == 'moviecorpus':
        collator_fn = Seq2SeqCollator(device='cpu')
    else:
        print("Given datasetname is not implemented!")
        raise NotImplementedError

    # --- make test loader ---
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE,
                             collate_fn=collator_fn)

    test_metrics = test(options, test_loader, model, vocab, DEVICE)
    print(test_metrics)
