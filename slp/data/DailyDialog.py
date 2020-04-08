import os
import numpy as np
import unicodedata
import re
import pickle
from torch.utils.data import Dataset
from slp.data.transforms import ToTensor

from slp.data.transforms import DialogSpacyTokenizer
from slp.config.special_tokens import DIALOG_SPECIAL_TOKENS
from slp.data.utils import make_train_val_test_split


class DailyDialogDataset(Dataset):
    def __init__(self, directory, transforms=None):

        self.transforms = transforms
        self.dialogues, self.emotions = self.read_dialogues(directory)
        self.triples, self.triples_labels = self.create_triples()
        self.word2count = {}

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
            pair[1]), self.normalizeString(
            pair[2])] for pair in self.triples]
        self.triples = norm_pairs

    def read_dialogues(self, directory):
        dialogues_file = os.path.join(directory, 'dialogues_text.txt')
        emotions_file = os.path.join(directory, 'dialogues_emotion.txt')

        dialogues = open(dialogues_file).readlines()
        emotions = open(emotions_file).readlines()

        dialogues = [dialog.strip().split('__eou__')[:-1] for dialog in
                     dialogues]
        emotions = [emotion.strip().split(' ') for emotion in emotions]

        keep_dialogues = []
        keep_emotions = []
        for dialog, emotion in zip(dialogues, emotions):
            if len(dialog) == len(emotion):
                keep_dialogues.append(dialog)
                keep_emotions.append(emotion)

        return keep_dialogues, keep_emotions

    def create_triples(self):
        triples = []
        triples_labels = []
        for dialog, emotion in zip(self.dialogues, self.emotions):
            if len(dialog) > 2:
                for i in range(len(dialog) - 2):
                    input1 = dialog[i].strip()
                    input2 = dialog[i+1].strip()
                    input3 = dialog[i+2].strip()
                    emotion1 = emotion[i]
                    emotion2 = emotion[i+1]
                    emotion3 = emotion[i+2]
                    if input1 and input2 and input3 and emotion1 and emotion2\
                            and emotion3:
                        triples.append([input1, input2, input3])
                        triples_labels.append([emotion1, emotion2, emotion3])

        return triples, triples_labels

    def threshold_data(self, max_length, tokenizer=None):
        keep = []
        for p in self.triples:
            if tokenizer is None:
                if (len(p[0].split(' ')) < max_length and len(p[1].split(' '))
                        < max_length and len(p[2].split(' '))<max_length ):
                    keep.append(p)
            else:
                if (len(tokenizer(p[0])) < max_length and len(tokenizer(p[1]))
                        < max_length and len(tokenizer(p[2])) < max_length ):
                    keep.append(p)
        self.triples = keep

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

    def trim_words(self, min_count, tokenizer=None):
        voc = self.create_vocab_dict(tokenizer)
        keep_pairs = []

        for triple in self.triples:
            input1_sentence = triple[0]
            input2_sentence = triple[1]
            output_sentence = triple[2]
            keep_input1 = True
            keep_input2 = True
            keep_output = True

            if tokenizer is None:
                for word in input1_sentence.split(' '):
                    if word not in voc or voc[word] < min_count:
                        keep_input1 = False
                        break
                for word in input2_sentence.split(' '):
                    if word not in voc or voc[word] < min_count:
                        keep_input2 = False
                        break
                for word in output_sentence.split(' '):
                    if word not in voc or voc[word] < min_count:
                        keep_output = False
                        break
            else:
                for word in tokenizer(input1_sentence):
                    if word not in voc or voc[word] < min_count:
                        keep_input1 = False
                        break
                for word in tokenizer(input2_sentence):
                    if word not in voc or voc[word] < min_count:
                        keep_input2 = False
                        break
                for word in tokenizer(output_sentence):
                    if word not in voc or voc[word] < min_count:
                        keep_output = False
                        break

            if keep_input1 and keep_input2 and keep_output:
                keep_pairs.append(triple)

        self.triples = keep_pairs
        self.word2count = self.create_vocab_dict(tokenizer)

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


class DailyDialogDatasetEmoTuples(Dataset):

    def __init__(self, directory, transforms=None):

        self.transforms = transforms
        self.dialogues, self.emotions = self.read_dialogues(directory)
        self.tuples, self.tuples_labels = self.create_tuples()
        self.word2count = {}

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

    def read_dialogues(self, directory):
        dialogues_file = os.path.join(directory, 'dialogues_text.txt')
        emotions_file = os.path.join(directory, 'dialogues_emotion.txt')

        dialogues = open(dialogues_file).readlines()
        emotions = open(emotions_file).readlines()

        dialogues = [dialog.strip().split('__eou__')[:-1] for dialog in
                     dialogues]
        emotions = [emotion.strip().split(' ') for emotion in emotions]

        keep_dialogues = []
        keep_emotions = []
        for dialog, emotion in zip(dialogues, emotions):
            if len(dialog) == len(emotion):
                keep_dialogues.append(dialog)
                keep_emotions.append(emotion)

        return keep_dialogues, keep_emotions

    def create_tuples(self):
        tuples = []
        tuples_labels = []
        for dialog, emotion in zip(self.dialogues, self.emotions):
            if len(dialog) > 1:
                for i in range(len(dialog) - 1):
                    input1 = dialog[i].strip()
                    input2 = dialog[i+1].strip()
                    emotion1 = emotion[i]
                    emotion2 = emotion[i+1]
                    if input1 and input2 and emotion1 and emotion2:
                        tuples.append([input1, input2])
                        tuples_labels.append([emotion1, emotion2])

        return tuples, tuples_labels

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

    def trim_words(self, min_count, tokenizer=None):
        voc = self.create_vocab_dict(tokenizer)
        keep_pairs = []

        for tuple in self.tuples:
            input1_sentence = tuple[0]
            input2_sentence = tuple[1]
            keep_input1 = True
            keep_input2 = True

            if tokenizer is None:
                for word in input1_sentence.split(' '):
                    if word not in voc or voc[word] < min_count:
                        keep_input1 = False
                        break
                for word in input2_sentence.split(' '):
                    if word not in voc or voc[word] < min_count:
                        keep_input2 = False
                        break
            else:
                for word in tokenizer(input1_sentence):
                    if word not in voc or voc[word] < min_count:
                        keep_input1 = False
                        break
                for word in tokenizer(input2_sentence):
                    if word not in voc or voc[word] < min_count:
                        keep_input2 = False
                        break
            if keep_input1 and keep_input2:
                keep_pairs.append(tuple)

        self.tuples = keep_pairs
        self.word2count = self.create_vocab_dict(tokenizer)

    def map(self, t):
        if self.transforms is None:
            self.transforms = []
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        s1, s2 = self.tuples[idx]
        emo1,emo2 = self.tuples_labels[idx]
        if self.transforms is not None:
            for t in self.transforms:
                s1 = t(s1)
                s2 = t(s2)

        tensor_conv = ToTensor()
        return s1, s2, tensor_conv(int(emo1)), tensor_conv(int(emo2))


class SubsetDailyDialogDatasetEmoTuples(Dataset):

    def __init__(self, train_list, transforms=None):

        self.transforms = transforms
        self.tuples, self.tuples_labels = self.read_list(train_list)
        self.word2count = {}

    def read_list(self,train_list):
        tuples = []
        tuples_labels = []
        for u1,u2, emo1,emo2 in train_list:

            tuples.append([u1, u2])
            tuples_labels.append([emo1,emo2])

        return tuples, tuples_labels

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
        emo1,emo2 = self.tuples_labels[idx]
        if self.transforms is not None:
            for t in self.transforms:
                s1 = t(s1)
                s2 = t(s2)
        return s1, s2, emo1, emo2

def make_pickles(outfolder):

    dataset = DailyDialogDatasetEmoTuples('./data/ijcnlp_dailydialog', transforms=None)
    print(dataset[0])
    tokenizer = DialogSpacyTokenizer(lower=True,
                                     specials=DIALOG_SPECIAL_TOKENS)
    dataset.threshold_data(15, tokenizer=tokenizer)
    dataset.trim_words(3, tokenizer=tokenizer)

    train_list,val_list,test_list = make_train_val_test_split(
        dataset, val_size=0.2, test_size=0.1)

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    with open(os.path.join(outfolder,'train_set.pkl'),'wb') as handle:
        pickle.dump(train_list,handle,protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(outfolder, 'val_set.pkl'), 'wb') as handle:
        pickle.dump(val_list,handle,protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(outfolder, 'test_set.pkl'), 'wb') as handle:
        pickle.dump(test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    make_pickles('./data/dailydialogpickles')