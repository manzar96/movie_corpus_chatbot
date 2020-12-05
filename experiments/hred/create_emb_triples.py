import argparse
from slp.config.special_tokens import HRED_SPECIAL_TOKENS
from slp.data.transforms import DialogSpacyTokenizer, ToTokenIds, ToTensor
from slp.util.embeddings import EmbeddingsLoader, create_emb_file
from slp.data.Semaine import SemaineDatasetTriplesOnly
from slp.data.moviecorpus import MovieCorpusDatasetTriples
from slp.data.DailyDialog import DailyDialogDataset



if __name__ == '__main__':
    # --- fix argument parser default values --
    parser = argparse.ArgumentParser(description='Main options')
    parser.add_argument('-dataset', type=str, help='Dataset used')
    parser.add_argument('-preprocess', action='store_true', default=False,
                        help='Preprocess dataset used')

    parser.add_argument('-embeddings', type=str, help='Embeddings file')
    parser.add_argument('-emb_dim', type=int, help='Embeddings dimension')

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
        dataset.threshold_data(13, tokenizer=tokenizer)
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
