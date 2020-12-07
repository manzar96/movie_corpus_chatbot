import argparse
import pickle
from tqdm import tqdm
from moverscore_v2 import get_idf_dict,word_mover_score
from collections import defaultdict
from moverscore_v2 import plot_example


def make_arg_parser():
    parser = argparse.ArgumentParser(description='Main options')
    parser.add_argument('-inpickle', type=str, help='Pickle file with '
                                                      'sentences',
                        required=True)
    parser.add_argument('-ngram', type=int, help='n grams for mover score',
                        required=True)
    parser.add_argument('-out', type=str, help='outputfolder to store '
                                                     'sentence embeddings',
                        default=None)
    parser.add_argument('-version', type=str, help='Model version to be used',
                        default='bert-base-nli-stsb-mean-tokens')
    return parser


if __name__ == "__main__":

    parser = make_arg_parser()
    options = parser.parse_args()

    with open(options.inpickle, 'rb') as handle:
        test_data = pickle.load(handle)
    handle.close()

    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    mover_scores = []

    for inp, answer, target in tqdm(test_data):
        translations = [answer]
        references = [target]

        scores = word_mover_score(references, translations, idf_dict_ref,
                                  idf_dict_hyp, stop_words=[],
                                  n_gram=options.ngram,
                                  remove_subwords=True)
        mover_scores.append(scores[0])

    print(sum(mover_scores)/len(mover_scores))
