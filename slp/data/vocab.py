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
