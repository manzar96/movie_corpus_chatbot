import os
import codecs
import csv
import unicodedata
import re
import numpy as np
from zipfile import ZipFile

from torch.utils.data import Dataset

from slp.config.moviecorpus import MOVIECORPUS_URL
from slp.util.system import download_url
from slp.data.transforms import *


class MovieCorpusDataset(Dataset):
    def __init__(self, directory, transforms=None, train=True):
        #dest = download_url(MOVIECORPUS_URL, directory)
        #with ZipFile(dest, 'r') as zipfd:
        #    zipfd.extractall(directory)
        self._file_lines = os.path.join(directory,
                                        'cornell '
                                        'movie-dialogs '
                                        'corpus',
                                        'movie_lines.txt')

        self._file_convs = os.path.join(directory,
                                        'cornell '
                                        'movie-dialogs '
                                        'corpus',
                                        'movie_conversations.txt')

        self.transforms = transforms
        self.pairs = self.get_metadata()
        self.transforms = transforms

    def get_metadata(self):
        movie_lines = open(self._file_lines, encoding='utf-8',
                           errors='ignore').read().split('\n')
        movie_conv_lines = open(self._file_convs, encoding='utf-8',
                                errors='ignore').read().split('\n')

        # Create a dictionary to map each line's id with its text
        id2line = {}
        for line in movie_lines:
            _line = line.split(' +++$+++ ')
            if len(_line) == 5:
                id2line[_line[0]] = _line[4]

        # Create a list of all of the conversations lines ids.
        convs = []
        for line in movie_conv_lines[:-1]:
            _line = line.split(' +++$+++ ')[-1][1:-1]\
                .replace("'", "").replace(" ", "")
            convs.append(_line.split(','))

        # Sort the sentences into questions (inputs) and answers (targets)
        questions = []
        answers = []

        for conv in convs:
            for i in range(len(conv) - 1):
                questions.append(id2line[conv[i]])
                answers.append(id2line[conv[i + 1]])
        return list(zip(questions, answers))

    def filter_data(self,min_threshold_length,max_threshold_length):
        """
        This method filters data according to threshold length.
        If the length of the turn exceeds threshold length the data is deleted.
        """
        spacytk = SpacyTokenizer()
        new_questions = []
        new_answers = []
        for question, answer in self.pairs:
            if max_threshold_length >= len(spacytk(question)) >= \
                    min_threshold_length and max_threshold_length >= \
                    len(spacytk(answer)) >= min_threshold_length:

                new_questions.append(question)
                new_answers.append(answer)
        
        self.pairs = list(zip(new_questions, new_answers))
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        question, answer = self.pairs[idx]

        if self.transforms is not None:
            question = self.transforms(question)
            answer = self.transforms(answer)

        return question, answer


class MovieCorpusDatasetv2(Dataset):
    def __init__(self, directory, transforms=None):
        # dest = download_url(MOVIECORPUS_URL, directory)
        # with ZipFile(dest, 'r') as zipfd:
        #    zipfd.extractall(directory)

        self.transforms = transforms
        new_dir = os.path.join(directory, 'cornell movie-dialogs corpus')
        self.file_lines = os.path.join(new_dir, 'movie_lines.txt')
        self.file_convs = os.path.join(new_dir, 'movie_conversations.txt')

        self.pairs = self.getdata(new_dir)
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

    def getdata(self, newdir):
        self.makeformattedfile(newdir)
        datafile = os.path.join(newdir, "formatted_movie_lines.txt")
        lines = open(datafile, encoding='utf-8'). \
            read().strip().split('\n')

        pairs = [[s for s in l.split('\t')] for l in lines]
        return pairs


        return s

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

    def word_counts(self, tokenizer):
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
        voc = self.word_counts(tokenizer)
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
                for word in tokenizer(input_sentence):
                    if word not in voc or voc[word]<min_count :
                        keep_output = False
                        break

            if keep_input and keep_output:
                keep_pairs.append(pair)

        self.pairs = keep_pairs
        self.word2count = self.word_counts(tokenizer)

    def apply_transforms(self, transforms):
        self.pairs = [[transforms(q), transforms(a)] for q, a in self.pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        question, answer = self.pairs[idx]

        if self.transforms is not None:
            question = self.transforms(question)
            answer = self.transforms(answer)

        return question, answer


if __name__ == '__main__':
    # from slp.util.embeddings import EmbeddingsLoader
    # from torchvision.transforms import Compose
    # loader = EmbeddingsLoader(
    #     './cache/glove.6B.50d.txt', 50)
    # word2idx, idx2word, embeddings = loader.load()
    #
    # transforms = Compose([SpacyTokenizer(), ToTokenIds(word2idx)])
    # data = MovieCorpusDataset('./data/', transforms=transforms)
    #
    # print(len(data))
    # data.filter_data(2, 13)
    # print(len(data))

    data2 = MovieCorpusDatasetv2('./data', transforms=None)
    data2.normalize_data()
    data2.threshold_data(10)
    data2.trim_words(3)
