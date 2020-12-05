import os
import pickle
import numpy as np

from torch.utils.data import Dataset

from slp.data.transforms import *


class MovieTriplesTrain(Dataset):

    def __init__(self, directory, transforms: list=None, append_eot=False):

        self.transforms = transforms
        self.directory = directory
        self.triples = self.read_data_triples()

    def read_data_triples(self):
        triples=[]
        with open(os.path.join(self.directory,
                               "MovieTriples/Training_Shuffled_Dataset.txt"))\
                as file:
            lines = file.readlines()
            for line in lines:
                input1, input2, target = line[:-1].split("\t")
                input1 = input1.strip()
                input2 = input2.strip()
                target = target.strip()
                triples.append([input1, input2, target])
        file.close()
        return triples

    def create_vocab_dict(self, tokenizer=None):
        voc_counts = {}
        for s1, s2, s3 in self.triples:
            if tokenizer is None:
                words, counts = np.unique(np.array(s1.split(' ')),
                                          return_counts=True)
            else:
                words, counts = np.unique(np.array(tokenizer(s1)),
                                          return_counts=True)
            for word, count in zip(words, counts):
                if word not in voc_counts.keys():
                    voc_counts[word] = count
                else:
                    voc_counts[word] += count

            if tokenizer is None:
                words, counts = np.unique(np.array(s2.split(' ')),
                                          return_counts=True)
            else:
                words, counts = np.unique(np.array(tokenizer(s2)),
                                          return_counts=True)
            for word, count in zip(words, counts):
                if word not in voc_counts.keys():
                    voc_counts[word] = count
                else:
                    voc_counts[word] += count

            if tokenizer is None:
                words, counts = np.unique(np.array(s3.split(' ')),
                                          return_counts=True)
            else:
                words, counts = np.unique(np.array(tokenizer(s3)),
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
        return len(self.triples)

    def __getitem__(self, idx):
        s1, s2, s3 = self.triples[idx]

        if self.transforms is not None:
            for t in self.transforms:
                s1 = t(s1)
                s2 = t(s2)
                s3 = t(s3)
        return s1, s2, s3


class MovieTriplesVal(Dataset):

    def __init__(self, directory, transforms: list = None):

        self.transforms = transforms
        self.directory = directory
        self.triples = self.read_data_triples()

    def read_data_triples(self):
        triples = []
        with open(os.path.join(self.directory,
                               "MovieTriples/Validation_Shuffled_Dataset.txt")) \
                as file:
            lines = file.readlines()
            for line in lines:
                input1, input2, target = line[:-1].split("\t")
                input1 = input1.strip()
                input2 = input2.strip()
                target = target.strip()
                triples.append([input1, input2, target])
        file.close()
        return triples

    def map(self, t):
        if self.transforms is None:
            self.transforms = []
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        s1, s2, s3 = self.triples[idx]

        if self.transforms is not None:
            for t in self.transforms:
                s1 = t(s1)
                s2 = t(s2)
                s3 = t(s3)
        return s1, s2, s3


class MovieTriplesTest(Dataset):

    def __init__(self, directory, transforms: list = None):

        self.transforms = transforms
        self.directory = directory
        self.triples = self.read_data_triples()

    def read_data_triples(self):
        triples = []
        with open(os.path.join(self.directory,
                               "MovieTriples/Test_Shuffled_Dataset.txt")) \
                as file:
            lines = file.readlines()
            for line in lines:
                input1, input2, target = line[:-1].split("\t")
                input1 = input1.strip()
                input2 = input2.strip()
                target = target.strip()
                triples.append([input1, input2, target])
        file.close()
        return triples

    def map(self, t):
        if self.transforms is None:
            self.transforms = []
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        s1, s2, s3 = self.triples[idx]

        if self.transforms is not None:
            for t in self.transforms:
                s1 = t(s1)
                s2 = t(s2)
                s3 = t(s3)
        return s1, s2, s3


if __name__ == '__main__':
    data = MovieTriplesTrain('./data', transforms=None)
    import ipdb;ipdb.set_trace()