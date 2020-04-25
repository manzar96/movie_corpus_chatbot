import torch
import argparse
import unicodedata
import re
import os
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from slp.data.transforms import DialogSpacyTokenizer, ToTokenIds, ToTensor
from slp.util import from_checkpoint
from slp.util.embeddings import EmbeddingsLoader
from slp.data.DailyDialog import SubsetDailyDialogDatasetEmoTuples
from slp.data.moviecorpus import SubsetMovieCorpusTuples
from slp.data.collators import NoEmoSeq2SeqCollator,Seq2SeqCollator
from slp.config.special_tokens import DIALOG_SPECIAL_TOKENS
from slp.modules.seq2seq.seq2seq import Encoder,Decoder,Seq2Seq
from slp.modules.seq2seq.inference import BeamSeq2Seq

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


def create_model(options, embeddings, emb_dim, vocab_size, sos_index,eos_index,
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

    model = Seq2Seq(encoder, decoder, sos_index, device,
                    shared_emb=options.shared_emb)
    return model


def create_searcher(options, model, beamsize, N_best, device):
    searcher = BeamSeq2Seq(model, options.maxseqlen, beam_size=beamsize,
                           N_best=N_best, device=device)
    return searcher


def load_embeddings(emb_file, emb_dim):
    loader = EmbeddingsLoader(emb_file, emb_dim,
                              extra_tokens=DIALOG_SPECIAL_TOKENS)
    word2idx, idx2word, embeddings = loader.load()
    return word2idx, idx2word, embeddings


def load_dataset_from_pickle(options):
    if os.path.exists(options.datasetfolder):
        if options.datasetname == 'dailydialog':
            with open(os.path.join(options.datasetfolder, 'train_set.pkl'),
                      'rb')as \
                    handle:
                test_list = pickle.load(handle)
                test_dataset = SubsetDailyDialogDatasetEmoTuples(test_list)
            handle.close()

        elif options.datasetname == 'moviecorpus':
            with open(os.path.join(options.datasetfolder, 'train_set.pkl'),
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


def test(outputfolder, test_loader, model, idx2word, device):

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    outfile = open(os.path.join(outputfolder, "test_results.txt"), "w")
    all_gen = []
    all_inputs = []
    all_targets = []
    generated_sent = []
    for index, batch in enumerate(tqdm(test_loader)):
        input1, lengths1, input2, lengths2 = batch
        input1 = input1.to(device)
        lengths1 = lengths1.to(device)
        outputs = model(input1,lengths1)
        for i in range(len(outputs)):
            generated_sent.append([idx2word[token] for token in outputs[i][0]])
        input_words = [idx2word[token.item()] for token in input1.squeeze(0)]
        golden_out_words = [idx2word[token.item()] for token in
                           input2.squeeze(0)]
        print_answers_to_file(outfile, generated_sent, input_words,
                              golden_out_words)
        all_inputs.append(input_words)
        all_targets.append(golden_out_words)
        all_gen.append(generated_sent)

    metrics = calc_metrics()
    print(metrics)

        # output_words = [idx2word[token.item()] for token in tokens]
        # input_words = [idx2word[token.item()] for token in input1.squeeze(0)]
        # golden_out_words = [idx2word[token.item()] for token in
        #                    input2.squeeze(0)]
        # outfile.write("Turn 1: "+str(input_words)+"\n")
        # outfile.write("Turn 2: "+str(output_words)+"\n")
        # outfile.write("Gold Turn 2: " + str(golden_out_words) + "\n")
        # outfile.write("================================================\n")
        # total += 1


def make_arg_parser():
    # --- fix argument parser default values --
    parser = argparse.ArgumentParser(description='Main options')

    parser.add_argument('-modelckpt', type=str, help='Model checkpoint',
                        required=True)
    parser.add_argument('-ckpt', type=str, help='checkpoint folder',
                        required=True)
    parser.add_argument('-output', type=str, help='output folder',
                        required=True)
    parser.add_argument('-datasetfolder', type=str,required=True,
                        help="Dataset folder path where pickle is saved")
    parser.add_argument('-datasetname', type=str, help='Dataset name',
                        required=True)

    parser.add_argument('-maxseqlen', type=int, default=15,
                        help='max seq len to be generated')

    parser.add_argument('-Nbest', type=int,default=1,help='number of responses')
    parser.add_argument('-beamsize', type=int,default=5, help='beamsearch size')

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

    vocab_size = len(word2idx)
    pad_index = word2idx[DIALOG_SPECIAL_TOKENS.PAD.value]
    sos_index = word2idx[DIALOG_SPECIAL_TOKENS.SOS.value]
    eos_index = word2idx[DIALOG_SPECIAL_TOKENS.EOS.value]
    print("Loaded Embeddings...")

    #  --- load model using loaded embeddings ---
    model = create_model(options, embeddings, options.emb_dim,
                         vocab_size, sos_index,eos_index, DEVICE)
    model = from_checkpoint(options.modelckpt, model, map_location='cpu')
    model = model.to(DEVICE)
    print("Loaded Model...")

    # --- create searcher for encoding user's input and for providing an
    # answer ---
    searcher = create_searcher(options, model, options.beamsize,
                               options.Nbest, DEVICE)
    searcher = searcher.to(DEVICE)
    searcher.eval()

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
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collator_fn)

    test(options.output, test_loader, searcher, idx2word, DEVICE)
