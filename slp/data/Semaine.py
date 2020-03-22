from torch.utils.data import Dataset
import glob
import os
import numpy as np
import unicodedata
import re


class SemaineDataset(Dataset):
    def __init__(self, directory, transforms=None, train=True):

        self.create_transcriptions(directory)
        self.create_datafile(directory)

    def create_datafile(self, directory):
        path = os.path.join(directory, "Sessions")
        for i in range(1, 129):
            new_path = os.path.join(path, str(i))
            transcription = os.path.join(new_path, 'transcription.txt')
            arousal_files = glob.glob(os.path.join(new_path, '*DA.txt'))
            valence_files = glob.glob(os.path.join(new_path, '*DV.txt'))

            #read labels and create line ,label file
            if os.path.exists(transcription):
                transcription = open(transcription,"r")
                trans_lines = transcription.readlines()
                next=1
                max = 0
                for line in trans_lines:
                    start, end,turn= line.split('\t')
                    start2, end2, turn2 = trans_lines[next].split('\t')
                    if int(start2)<int(end):
                        if int(end)-int(start2) > max:
                            max = int(end)-int(start2)
                            max_start=start
                        next+=1
                        if(next==len(trans_lines)-1):
                            break
                print(i,max,max_start)
    def create_transcriptions(self, directory):
        path = os.path.join(directory, "Sessions")
        for i in range(1, 15):
            new_path = os.path.join(path, str(i))
            transcript = glob.glob(os.path.join(new_path,
                                                 'alignedTranscript_*.txt'))
            word_level_user = glob.glob(os.path.join(new_path,
                                                     'wordLevel*user*'))
            word_level_operator = glob.glob(os.path.join(
                new_path, 'wordLevel*operator*'))

            arousal_files = glob.glob(os.path.join(new_path, '*DA.txt'))
            valence_files = glob.glob(os.path.join(new_path, '*DV.txt'))


            if not transcript == []:
                trans = open(os.path.join(new_path, "transcription.txt"), "w")
                transcript = open(transcript[0], "r")
                word_user = open(word_level_user[0], "r")
                word_operator = open(word_level_operator[0], "r")

                t_lines = transcript.readlines()
                user_lines = word_user.readlines()
                operator_lines = word_operator.readlines()
                counter_user_lines = 0
                counter_operator_lines = 0

                for line in t_lines:
                    new=True
                    if line.split("\t")[0].split(" ")[2] == "User":
                        if counter_user_lines < len(user_lines):
                            if user_lines[counter_user_lines][0]=="-":
                                counter_user_lines += 1

                            while not user_lines[counter_user_lines][
                                          0]=="-":
                                if new and user_lines[
                                    counter_user_lines][0] ==".":
                                    time_start="empty"
                                    time_end = "empty"
                                    print("empty")
                                    new=False
                                    
                                elif new and not user_lines[ 
                                                     counter_user_lines][
                                                     0] ==".":
                                    time_start=user_lines[
                                        counter_user_lines].split(" ")[0]
                                    time_end = user_lines[
                                        counter_user_lines].split(" ")[1]
                                    new = False
                                
                                else:
                                    if not user_lines[counter_user_lines][
                                        0] == ".":
                                        time_end = user_lines[
                                            counter_user_lines].split(" ")[
                                            1]
                                        
                                counter_user_lines += 1
                                if counter_user_lines==len(user_lines):
                                    break
                            trans.write((time_start + "\t" + time_end + "\t" +
                                         line.split(
                                             '\t')[1]))                                            
                    else:
                        if counter_operator_lines < len(operator_lines):
                            if operator_lines[counter_operator_lines][0] == "-":
                                counter_operator_lines += 1

                            while not operator_lines[counter_operator_lines][
                                          0] == "-":
                                if new and operator_lines[
                                    counter_operator_lines][0] == ".":
                                    time_start = "empty"
                                    time_end = "empty"
                                    print("empty")
                                    new = False

                                elif new and not operator_lines[
                                                     counter_operator_lines][
                                                     0] == ".":
                                    time_start = operator_lines[
                                        counter_operator_lines].split(" ")[0]
                                    time_end = operator_lines[
                                        counter_operator_lines].split(" ")[1]
                                    new = False

                                else:
                                    if not operator_lines[counter_operator_lines][
                                               0] == ".":
                                        time_end = operator_lines[
                                            counter_operator_lines].split(" ")[
                                            1]

                                counter_operator_lines += 1
                                if counter_operator_lines == len(
                                        operator_lines):
                                    break
                            trans.write((time_start + "\t" + time_end + "\t" +
                                         line.split(
                                             '\t')[1]))

                trans.close()
                transcript.close()
                word_user.close()
                word_operator.close()
            # line_label_counter = 0
            # if not arousal_files == []:
            #     label_file = open(arousal_files[0], "r")
            #     labels_lines = label_file.readlines()
            # else:
            #     pass
            #
            # if not transcript == []:
            #     with open(transcript[0],"r")as tfile:
            #
            #         lines = tfile.readlines()
            #         for index,line in enumerate(lines):
            #
            #             start_time = line.split("\t")[0].split(" ")[0]
            #             end_time = line.split("\t")[0].split(" ")[1]
            #             utt1 = line.split("\t")[1]
            #             utt1 = re.sub("\([^)]+\).","",utt1)
            #             utt1 = utt1.strip()
            #
            #             start_min,start_sec,start_ms = start_time.split(":")
            #             start1 = int(start_min)*60+int(
            #                 start_sec)+int(start_ms)/1000
            #             end_min,end_sec,end_ms = end_time.split(":")
            #             end1 = int(end_min)*60+int(
            #                 end_sec)+int(end_ms)/1000
            #             print(utt1)
            #
            #             #stop writing to file as we read all lines!
            #             if index < len(lines)-2:
            #                 next_line = lines[index+1]
            #             else:
            #                 break
            #
            #             start_time = next_line.split("\t")[0].split(" ")[0]
            #             end_time = next_line.split("\t")[0].split(" ")[1]
            #             utt2 = next_line.split("\t")[1]
            #             utt2 = re.sub("\([^)]+\).","",utt2)
            #             utt2 = utt2.strip()
            #
            #             start_min,start_sec,start_ms = start_time.split(":")
            #             start2 = int(start_min)*60+int(
            #                 start_sec)+int(start_ms)/1000
            #             end_min,end_sec,end_ms = end_time.split(":")
            #             end2 = int(end_min)*60+int(
            #                 end_sec)+int(end_ms)/1000
            #             print(utt1)
            #
            #             while line_label_counter < len(labels_lines):
            #                 label_line = labels_lines[line_label_counter]
            #                 time, label = label_line.strip().split(" ")
            #                 time = float(time)
            #                 if start1+time < end2:
            #                     print(time, label)
            #
            #                 line_label_counter += 1



    def map(self, t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        id = self.ids[idx]
        # print(id)
        s1, s2, s3 = self.triples[idx]
        if self.transforms is not None:
            for t in self.transforms:
                s1 = t(s1)
                s2 = t(s2)
                s3 = t(s3)

        return s1, s2, s3


class SemaineDatasetTriplesOnly(Dataset):
    def __init__(self, directory, transforms=None, train=True):

        self.triples = self.create_triples(directory)
        self.transforms = transforms

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

    def create_triples(self, directory):
        path = os.path.join(directory, "Sessions")
        triples=[]
        for i in range(1, 129):
            new_path = os.path.join(path, str(i))
            transcript = glob.glob(os.path.join(new_path,"aligned*.txt"))
            if not transcript == []:
                trans = open(transcript[0], "r")
                trans_lines = trans.readlines()

                for i in range(len(trans_lines)-2):
                    u1 = trans_lines[i].strip().split("\t")[1]
                    u2 = trans_lines[i+1].strip().split("\t")[1]
                    u3 = trans_lines[i+2].strip().split("\t")[1]
                    triples.append([u1,u2,u3])

        return triples

    def map(self, t):
        if self.transforms is None:
            self.transforms = []
        self.transforms.append(t)
        return self

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
    dataset = SemaineDatasetTriplesOnly(
        "./data/semaine-database_download_2020-01-21_11_41_49")
    import ipdb;ipdb.set_trace()