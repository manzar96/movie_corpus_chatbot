from torch.utils.data import Dataset
import numpy as np
import unicodedata
import re


class SubTle(Dataset):
    def __init__(self, directory, samples_limit=None, transforms=None):

        self.limit = samples_limit
        self.ids, self.tuples = self.read_data(directory)
        self.transforms = transforms
        self.word2count = {}
    def read_data(self, directory):
        if self.limit is None:
            lines = open(directory).read().split("\n")[:-1]
        else:
            lines = open(directory).read().split("\n")[:-1][:(self.limit+1)*6]
        tuples = []
        ids = []
        for index, line in enumerate(lines):
            if not line == '':
                if line[0] == 'S':
                    ids.append(line.split()[2])
                elif line[0] == 'I':
                    u1 = line[4:]
                elif line[0] == 'R':
                    u2 = line[4:]

            if index % 6 == 0 and line == '' and (not index == 0):
                tuples.append((u1, u2))
                
                if self.limit is not None:
                    if len(tuples)>self.limit:
                        break

        return ids, tuples

    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def normalize_data(self):
        norm_pairs = [[self.normalizeString(pair[0]), self.normalizeString(
            pair[1])] for pair in self.tuples]
        self.tuples = norm_pairs

    def threshold_data(self, max_length, tokenizer=None):
        keep = []
        for p in self.tuples:
            if tokenizer is None:
                if (len(p[0].split(' ')) < max_length and len(p[1].split(' '))
                        < max_length):
                    keep.append(p)
            else:
                if (len(tokenizer(p[0])) < max_length and len(tokenizer(p[1]))
                        < max_length):
                    keep.append(p)
        self.tuples = keep

    def word_counts(self, tokenizer):
        voc_counts = {}
        for question, answer in self.tuples:
            if tokenizer is None:
                words, counts = np.unique(np.array(question.split(' ')),
                                          return_counts=True)
            else:
                words, counts = np.unique(np.array(tokenizer(question)),
                                          return_counts=True)
            for word, count in zip(words, counts):
                if word not in voc_counts.keys():
                    voc_counts[word] = count
                else:
                    voc_counts[word] += count

            if tokenizer is None:
                words, counts = np.unique(np.array(answer.split(' ')),
                                          return_counts=True)
            else:
                words, counts = np.unique(np.array(tokenizer(answer)),
                                          return_counts=True)
            for word, count in zip(words, counts):
                if word not in voc_counts.keys():
                    voc_counts[word] = count
                else:
                    voc_counts[word] += count

        return voc_counts

    def trim_words(self, min_count, tokenizer=None):
        voc = self.word_counts(tokenizer)
        keep_tuples = []

        for pair in self.tuples:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True

            # Check input sentence
            if tokenizer is None:
                for word in input_sentence.split(' '):
                    if word not in voc or voc[word] < min_count:
                        keep_input = False
                        break
                # Check output sentence
                for word in output_sentence.split(' '):
                    if word not in voc or voc[word] < min_count:
                        keep_output = False
                        break
            else:
                for word in tokenizer(input_sentence):
                    if word not in voc or voc[word] < min_count:
                        keep_input = False
                        break
                # Check output sentence
                for word in tokenizer(input_sentence):
                    if word not in voc or voc[word] < min_count:
                        keep_output = False
                        break

            if keep_input and keep_output:
                keep_tuples.append(pair)

        self.tuples = keep_tuples
        self.word2count = self.word_counts(tokenizer)

    def create_vocab_dict(self, tokenizer):
        """
        receives a tokenizer in order to split sentences and create
        a dict-vocabulary with words counts.
        """

        voc_counts = {}
        for s1, s2 in self.tuples:
            words, counts = np.unique(np.array(tokenizer(s1)),
                                      return_counts=True)
            for word, count in zip(words, counts):
                if word not in voc_counts.keys():
                    voc_counts[word] = count
                else:
                    voc_counts[word] += count

            words, counts = np.unique(np.array(tokenizer(s2)),
                                      return_counts=True)
            for word, count in zip(words, counts):
                if word not in voc_counts.keys():
                    voc_counts[word] = count
                else:
                    voc_counts[word] += count
        return voc_counts

    def map(self, t):
        if self.transforms is None:
            self.transforms = []
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        id = self.ids[idx]

        s1, s2 = self.tuples[idx]
        if self.transforms is not None:
            for t in self.transforms:
                s1 = t(s1)
                s2 = t(s2)
        return s1, s2


class SubTriples2(Dataset):
    def __init__(self, directory, transforms=None, train=True):

        self.ids, self.triples = self.read_data(directory)
        self.transforms = transforms

    def read_data(self, directory):
        lines = open(directory).read().split("\n")[:-1]
        triplets = []
        ids = []
        for index, line in enumerate(lines):
            if not line == '':
                if line[0] == 'S':
                    ids.append(line.split()[2])
                elif line[0] == 'I':
                    u2 = line[4:]
                elif line[0] == 'R':
                    u3 = line[4:]

            if index % 6 == 0 and line == '' and (not index == 0):
                triplets.append(('', u2, u3))

        return ids, triplets

    def map(self, t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        id = self.ids[idx]
        s1, s2, s3 = self.triples[idx]
        if self.transforms is not None:
            for t in self.transforms:
                s1 = t(s1)
                s2 = t(s2)
                s3 = t(s3)

        return s1, s2, s3

if __name__ == '__main__':
    dataset = SubTle('/home/manzar/Downloads/DATASETS/SubtleCorpusPTEN/eng/corpus0sDialogues.txt')
    import ipdb;ipdb.set_trace()


