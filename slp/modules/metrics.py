import re
import Levenshtein as Lev
from itertools import chain
from nltk.translate import bleu_score



def calc_sentence_bleu_score(reference,hypothesis,n=4):
    """
    This function receives as input a reference sentence(list) and a
    hypothesis(list) and
    returns bleu score.

    Bleu score formula: https://leimao.github.io/blog/BLEU-Score/
    """
    reference = [reference]
    weights = [1/n for _ in range(n)]
    return bleu_score.sentence_bleu(reference, hypothesis, weights)


def calc_word_error_rate(s1,s2):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences (list of words should be given).
    s1:reference
    s2:hypothesis
    """

    # build mapping of words to integers
    ba = set(s1+ s2)
    word2idx = dict(zip(ba, range(len(ba))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2idx[w]) for w in s1]
    w2 = [chr(word2idx[w]) for w in s2]
    return Lev.distance(''.join(w1), ''.join(w2)) / len(s1)

import torch
import torch.nn as nn

import torch.nn.functional as F
class BLEU_metric_generation(nn.Module):
    def __init__(self, batch_size, pad_idx=0, idx2word={}):
        super(BLEU_metric_generation,
              self).__init__()
        self.pad_idx = pad_idx
        self.batch_size = batch_size
        self.idx2word = idx2word

    def forward(self, y_pred, targets):
        targets = targets.reshape(self.batch_size, -1)
        voc_dim = y_pred.shape[1]
        y_pred = y_pred.reshape(self.batch_size, -1, voc_dim)
        all_bleu_1 = []
        all_bleu_2 = []
        all_bleu_3 = []
        all_bleu_4 = []
        for i in range(self.batch_size):
            prediction = y_pred[i]
            top_index = F.log_softmax(prediction, dim=1)
            _, topi = top_index.topk(1, dim=-1)
            target = targets[i]
            ziped = [(pred, tgt) for pred, tgt in zip(topi, target) if \
                     tgt.item() != self.pad_idx]
            preds_idx = [pred.item() for pred, tgt in ziped]
            tgt_idx = [tgt.item() for pred, tgt in ziped]
            pred_words = [self.idx2word[pred] for pred in preds_idx]
            tgt_words = [self.idx2word[tgt] for tgt in tgt_idx]
            bleu1 = calc_sentence_bleu_score(tgt_words,pred_words,n=1)
            bleu2 = calc_sentence_bleu_score(tgt_words,pred_words,n=2)
            all_bleu_1.append(bleu1)
            all_bleu_2.append(bleu2)

        return sum(all_bleu_2) / len(all_bleu_2)


class BleuMetric:
    """
    According to ParlAI
    """
    def __init__(self):
        self.re_art = re.compile(r'\b(a|an|the)\b')
        self.re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

    def normalize_answer(self,s):
        """
        Lower text and remove punctuation, articles and extra whitespace.
        """

        s = s.lower()
        s = self.re_punc.sub(' ', s)
        s = self.re_art.sub(' ', s)
        # TODO: this could almost certainly be faster with a regex \s+ -> ' '
        s = ' '.join(s.split())
        return s

    def compute(self, answers, reference, k):
        """
        Compute approximate BLEU score between reference a set of answers.
        :param answers: list of candidate answers ['Hi how are you','Let's go
        now']
        :param reference: reference sentence
        :param k: bleu k-gram to be computed
        :return: score
        """
        weights = [1 / k for _ in range(k)]
        score = bleu_score.sentence_bleu(
            [self.normalize_answer(a).split(" ") for a in answers],
            self.normalize_answer(reference).split(" "),
            smoothing_function=bleu_score.SmoothingFunction(epsilon=1e-12).method1,
            weights=weights,
        )
        return score


class DistinctN:
    """
    Copied from https://github.com/neural-dialogue-metrics/Distinct-N
    """
    def __init__(self, pad_left=False, pad_right=False,
                     left_pad_symbol=None, right_pad_symbol=None):
        """

        :param pad_left: whether the ngrams should be left-padded
        :param pad_right: whether the ngrams should be right-padded
        :param left_pad_symbol: the symbol to use for left padding (default is None)
        :param right_pad_symbol: the symbol to use for right padding (default is None)
        """
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.left_pad_symbol = left_pad_symbol
        self.right_pad_symbol = right_pad_symbol
        self.re_art = re.compile(r'\b(a|an|the)\b')
        self.re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

    def pad_sequence(self, sequence, n):
        """
        Returns a padded sequence of items before ngram extraction.
            # >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
            ['<s>', 1, 2, 3, 4, 5, '</s>']
            # >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
            ['<s>', 1, 2, 3, 4, 5]
            # >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
            [1, 2, 3, 4, 5, '</s>']
        :param sequence: the source data to be padded
        :type sequence: sequence or iter
        :param n: the degree of the ngrams
        :type n: int
        :rtype: sequence or iter
        """
        sequence = iter(sequence)
        if self.pad_left:
            sequence = chain((self.left_pad_symbol,) * (n - 1), sequence)
        if self.pad_right:
            sequence = chain(sequence, (self.right_pad_symbol,) * (n - 1))
        return sequence

    def ngrams(self, sequence, n):
        """
        Return the ngrams generated from a sequence of items, as an iterator.
        For example:
            # >>> from nltk.util import ngrams
            # >>> list(ngrams([1,2,3,4,5], 3))
            [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
        Wrap with list for a list version of this function.  Set pad_left
        or pad_right to true in order to get additional ngrams:
            # >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
            # >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
            # >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
            [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
            # >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
            [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        :param sequence: the source data to be converted into ngrams
        :type sequence: sequence or iter
        :param n: the degree of the ngrams
        :type n: int
        :rtype: sequence or iter
        """
        sequence = self.pad_sequence(sequence, n)
        history = []
        while n > 1:
            history.append(next(sequence))
            n -= 1
        for item in sequence:
            history.append(item)
            yield tuple(history)
            del history[0]

    def normalize_answer(self, s):
        """
        Lower text and remove punctuation, articles and extra whitespace.
        """

        s = s.lower()
        s = self.re_punc.sub(' ', s)
        s = self.re_art.sub(' ', s)
        # TODO: this could almost certainly be faster with a regex \s+ -> ' '
        s = ' '.join(s.split())
        return s

    def distinct_n_sentence_level(self, sentence, n):
        """
        Compute distinct-N for a single sentence.
        :param sentence: a sentence in which we compute n-gram
        :param n: int, ngram.
        :return: float, the metric value.
        """
        sentence = self.normalize_answer(sentence).split(" ")
        if len(sentence) == 0:
            return 0.0  # Prevent a zero division
        distinct_ngrams = set(self.ngrams(sentence, n))
        return len(distinct_ngrams) / len(sentence)


if __name__=="__main__":
    ref = ["hi","how","are","you","today","?"]
    hyp = ["I","am","fine","."]
    print(calc_sentence_bleu_score(ref,hyp,4))