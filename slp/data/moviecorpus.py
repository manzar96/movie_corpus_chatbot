import os
import codecs
import csv
import unicodedata
import re
import numpy as np
import pickle
from zipfile import ZipFile

from torch.utils.data import Dataset

from slp.config.moviecorpus import MOVIECORPUS_URL
from slp.util.system import download_url

from slp.data.transforms import DialogSpacyTokenizer
from slp.config.special_tokens import DIALOG_SPECIAL_TOKENS
from slp.data.utils import make_train_val_test_split

class MovieCorpusDatasetTuples(Dataset):
    def __init__(self, directory, transforms=None):
        dest = download_url(MOVIECORPUS_URL, directory)
        with ZipFile(dest, 'r') as zipfd:
           zipfd.extractall(directory)

        self.transforms = transforms
        new_dir = os.path.join(directory, 'cornell movie-dialogs corpus')
        self.file_lines = os.path.join(new_dir, 'movie_lines.txt')
        self.file_convs = os.path.join(new_dir, 'movie_conversations.txt')

        self.pairs = self.getdata()
        self.word2count={}

    def load_lines(self, filename, fields):
        lines = {}
        with open(filename, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]
                lines[lineObj['lineID']] = lineObj
        return lines

    def load_conversations(self, filename, lines, fields):
        conversations = []
        with open(filename, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]
                # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
                lineIds = eval(convObj["utteranceIDs"])
                # Reassemble lines
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(lines[lineId])
                conversations.append(convObj)
        return conversations

    def extract_sentencepairs(self, conversations):
        qa_pairs = []
        for conversation in conversations:
            # Iterate over all the lines of the conversation
            for i in range(len(conversation[
                                   "lines"]) - 1):  # We ignore the last line (no answer for it)
                inputLine = conversation["lines"][i]["text"].strip()
                targetLine = conversation["lines"][i + 1]["text"].strip()
                # Filter wrong samples (if one of the lists is empty)
                if inputLine and targetLine:
                    qa_pairs.append([inputLine, targetLine])
        return qa_pairs

    def makeformattedfile(self, new_dir):
        # Initialize lines dict, conversations list, and field ids
        lines = {}
        conversations = []
        MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character",
                              "text"]
        MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID",
                                      "utteranceIDs"]

        lines = self.load_lines(self.file_lines, MOVIE_LINES_FIELDS)
        conversations = self.load_conversations(self.file_convs, lines,
                                               MOVIE_CONVERSATIONS_FIELDS)

        datafile = os.path.join(new_dir, "formatted_movie_lines.txt")
        delimiter = '\t'

        # Unescape the delimiter
        delimiter = str(codecs.decode(delimiter, "unicode_escape"))

        with open(datafile, 'w', encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter=delimiter)
            for pair in self.extract_sentencepairs(conversations):
                writer.writerow(pair)

    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalizeString(self,s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def getdata(self):
        MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character",
                              "text"]
        MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID",
                                      "utteranceIDs"]

        lines = self.load_lines(self.file_lines, MOVIE_LINES_FIELDS)
        conversations = self.load_conversations(self.file_convs, lines,
                                               MOVIE_CONVERSATIONS_FIELDS)
        pairs = [(q.strip(), a.strip()) for q, a in self.extract_sentencepairs(
                conversations)]
        return pairs

    def normalize_data(self):
        norm_pairs = [[self.normalizeString(pair[0]),self.normalizeString(
            pair[1])] for pair in self.pairs]
        self.pairs=norm_pairs

    def threshold_data(self, max_length, tokenizer=None):
        keep=[]
        for p in self.pairs:
            if tokenizer is None:
                if (len(p[0].split(' ')) < max_length and len(p[1].split(' '))
                        < max_length):
                    keep.append(p)
            else:
                if (len(tokenizer(p[0])) < max_length and len(tokenizer(p[1]))
                        < max_length):
                    keep.append(p)
        self.pairs = keep

    def create_vocab_dict(self, tokenizer):
        voc_counts = {}
        for question, answer in self.pairs:
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
        voc = self.create_vocab_dict(tokenizer)
        keep_pairs = []

        for pair in self.pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True

            # Check input sentence
            if tokenizer is None:
                for word in input_sentence.split(' '):
                    if word not in voc or voc[word]<min_count :
                        keep_input = False
                        break
                # Check output sentence
                for word in output_sentence.split(' '):
                    if word not in voc or voc[word]<min_count :
                        keep_output = False
                        break
            else:
                for word in tokenizer(input_sentence):
                    if word not in voc or voc[word]<min_count :
                        keep_input = False
                        break
                # Check output sentence
                for word in tokenizer(output_sentence):
                    if word not in voc or voc[word]<min_count :
                        keep_output = False
                        break

            if keep_input and keep_output:
                keep_pairs.append(pair)

        self.pairs = keep_pairs
        self.word2count = self.create_vocab_dict(tokenizer)

    def map(self, t):
        if self.transforms is None:
            self.transforms = []
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s1, s2 = self.pairs[idx]
        if self.transforms is not None:
            for t in self.transforms:
                s1 = t(s1)
                s2 = t(s2)
        return s1, s2



class SubsetMovieCorpusTuples(Dataset):

    def __init__(self, train_list, transforms=None):

        self.transforms = transforms
        self.tuples = train_list
        self.word2count = {}

    def create_vocab_dict(self, tokenizer=None):
        voc_counts = {}
        for s1, s2 in self.tuples:
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
        return voc_counts

    def map(self, t):
        if self.transforms is None:
            self.transforms = []
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        s1, s2 = self.tuples[idx]
        if self.transforms is not None:
            for t in self.transforms:
                s1 = t(s1)
                s2 = t(s2)
        return s1, s2

def make_pickles(outfolder):

    dataset = MovieCorpusDatasetTuples('./data', transforms=None)
    print(dataset[0])
    tokenizer = DialogSpacyTokenizer(lower=True,
                                     specials=DIALOG_SPECIAL_TOKENS)
    dataset.threshold_data(20, tokenizer=tokenizer)
    #dataset.trim_words(3, tokenizer=tokenizer)
    train_list,val_list,test_list = make_train_val_test_split(
        dataset, val_size=0.01, test_size=0.05)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    with open(os.path.join(outfolder,'train_set.pkl'),'wb') as handle:
        pickle.dump(train_list,handle,protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(outfolder, 'val_set.pkl'), 'wb') as handle:
        pickle.dump(val_list,handle,protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(outfolder, 'test_set.pkl'), 'wb') as handle:
        pickle.dump(test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    make_pickles('./data/pickles/moviecorpus')
