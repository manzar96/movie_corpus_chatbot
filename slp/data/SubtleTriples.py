from torch.utils.data import Dataset
import numpy as np


class SubTle(Dataset):
    def __init__(self, directory, transforms=None, train=True):

        self.ids, self.tuples = self.read_data(directory)
        self.transforms = transforms

    def read_data(self, directory):
        lines = open(directory).read().split("\n")[:-1][:1000000]
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

        return ids, tuples

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


