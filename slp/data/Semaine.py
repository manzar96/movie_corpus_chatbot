from torch.utils.data import Dataset
import glob
import os
import re


class SemaineDataset(Dataset):
    def __init__(self, directory, transforms=None, train=True):

        self.read_data(directory)

    def read_data(self, directory):
        path = os.path.join(directory, "Sessions")
        for i in range(1, 3):
            new_path = os.path.join(path, str(i))
            transcript = glob.glob(os.path.join(new_path,
                                                 'alignedTranscript_*.txt'))
            word_level_user = glob.glob(os.path.join(new_path,
                                                     'wordLevel*user'))
            word_level_operator = glob.glob(os.path.join(
                new_path, 'wordLevel*operator'))
            arousal_files = glob.glob(os.path.join(new_path, '*DA.txt'))

            if not transcript == []:
                trans = open(os.path.join(new_path, "transcription.txt"), "w")
                transcript = open(transcript,"r")
                word_user = open(word_level_user,"r")
                word_level_operator = open(word_level_operator,"r")


            line_label_counter = 0
            if not arousal_files == []:
                label_file = open(arousal_files[0], "r")
                labels_lines = label_file.readlines()
            else:
                pass

            if not transcript == []:
                with open(transcript[0],"r")as tfile:

                    lines = tfile.readlines()
                    for index,line in enumerate(lines):

                        start_time = line.split("\t")[0].split(" ")[0]
                        end_time = line.split("\t")[0].split(" ")[1]
                        utt1 = line.split("\t")[1]
                        utt1 = re.sub("\([^)]+\).","",utt1)
                        utt1 = utt1.strip()

                        start_min,start_sec,start_ms = start_time.split(":")
                        start1 = int(start_min)*60+int(
                            start_sec)+int(start_ms)/1000
                        end_min,end_sec,end_ms = end_time.split(":")
                        end1 = int(end_min)*60+int(
                            end_sec)+int(end_ms)/1000
                        print(utt1)

                        #stop writing to file as we read all lines!
                        if index < len(lines)-2:
                            next_line = lines[index+1]
                        else:
                            break

                        start_time = next_line.split("\t")[0].split(" ")[0]
                        end_time = next_line.split("\t")[0].split(" ")[1]
                        utt2 = next_line.split("\t")[1]
                        utt2 = re.sub("\([^)]+\).","",utt2)
                        utt2 = utt2.strip()

                        start_min,start_sec,start_ms = start_time.split(":")
                        start2 = int(start_min)*60+int(
                            start_sec)+int(start_ms)/1000
                        end_min,end_sec,end_ms = end_time.split(":")
                        end2 = int(end_min)*60+int(
                            end_sec)+int(end_ms)/1000
                        print(utt1)

                        while line_label_counter < len(labels_lines):
                            label_line = labels_lines[line_label_counter]
                            time, label = label_line.strip().split(" ")
                            time = float(time)
                            if start1+time < end2:
                                print(time, label)

                            line_label_counter += 1



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

if __name__=='__main__':
    SemaineDataset("./data/semaine-database_download_2020-01-21_11_41_49")