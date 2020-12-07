import itertools
from collections import Counter


def create_vocab(corpus, vocab_size=5000, extra_tokens=None):
    if isinstance(corpus[0], list):
        corpus = itertools.chain.from_iterable(corpus)
    freq = Counter(corpus)
    if extra_tokens is None:
        extra_tokens = []
    take = min(vocab_size, len(freq))
    common_words = list(map(lambda x: x[0], freq.most_common(take)))
    common_words = list(set(common_words) - set(extra_tokens))
    words = extra_tokens + common_words
    if len(words) > vocab_size:
        words = words[:vocab_size]
    vocab = dict(zip(words, itertools.count()))
    return vocab


def word2idx_from_dataset(wordcounts, most_freq=None, extra_tokens=None):
    word2idx = {}
    idx2word = {}
    counter = 0
    if extra_tokens is not None:
        for token in extra_tokens:
            word2idx[token.value] = counter
            idx2word[counter] = token.value
            counter += 1
    if most_freq is None:
        for word in wordcounts:
            if word not in word2idx:
                word2idx[word] = counter
                idx2word[counter] = word
                counter += 1
    else:
        sorted_voc = sorted(wordcounts.items(), key=lambda kv: kv[1])
        for word in sorted_voc[-most_freq:]:
            if word[0] not in word2idx:
                word2idx[word[0]] = counter
                idx2word[counter] = word[0]
                counter += 1
    return word2idx, idx2word

def tensor2text(vec,vocab):
    """
    converts vec to txt! padding values are erased!
    :param vec:
    :param vocab:
    :return:
    """
    new_vec = []
    for i in vec:
        if i == vocab.end_idx:
            break
        elif i != vocab.start_idx:
            new_vec.append(i)
    words = [vocab.idx2word[idx.item()] for idx in new_vec]
    text = " ".join(words)
    return text


class Vocab:
    def __init__(self, word2idx, idx2word, extra_tokens):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.extra_tokens = extra_tokens
        self.start_idx = word2idx[extra_tokens.SOS.value]
        self.end_idx = word2idx[extra_tokens.EOS.value]
        self.unk_idx = word2idx[extra_tokens.UNK.value]
        self.pad_idx = word2idx[extra_tokens.PAD.value]

    def __len__(self):
        return len(self.word2idx)

