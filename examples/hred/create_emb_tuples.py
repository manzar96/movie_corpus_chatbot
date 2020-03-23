import argparse

from slp.config.special_tokens import HRED_SPECIAL_TOKENS
from slp.data.transforms import DialogSpacyTokenizer, ToTokenIds, ToTensor
from slp.data.Subtle import SubTle
from slp.data.moviecorpus import MovieCorpusDatasetv2
from slp.util.embeddings import EmbeddingsLoader, create_emb_file


if __name__ == '__main__':

    # --- fix argument parser default values --
    parser = argparse.ArgumentParser(description='Main options')
    parser.add_argument('-dataset', type=str, help='Dataset used')
    parser.add_argument('-preprocess', action='store_true', default=False,
                        help='Preprocess dataset used')

    parser.add_argument('-embeddings', type=str, help='Embeddings file')
    parser.add_argument('-emb_dim', type=int, help='Embeddings dimension')

    parser.add_argument('-sl', dest='samplelimit', type=int,
                        default=100000, help='sample limit used for training')


    options = parser.parse_args()

    # ---  read data to create vocabulary dict ---
    tokenizer = DialogSpacyTokenizer(lower=True,
                                     specials=HRED_SPECIAL_TOKENS)
    if options.dataset == "movie":
        dataset = MovieCorpusDatasetv2('./data/', transforms=None)
    elif options.dataset == "subtle":
        dataset = SubTle("./data/corpus0sDialogues.txt",
                         samples_limit=options.samplelimit, transforms=None)
    else:
        assert False, "Specify dataset used in options (movie or subtle)"

    dataset.normalize_data()
    if options.preprocess:
        dataset.threshold_data(10, tokenizer=tokenizer)
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

