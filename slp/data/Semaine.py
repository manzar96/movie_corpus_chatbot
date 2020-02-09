from torch.utils.data import Dataset
import glob
import os
import re


class SemaineDataset(Dataset):
    def __init__(self, directory, transforms=None, train=True):

        self.create_transcriptions(directory)
        self.create_datafile(directory)

    def create_datafile(self, directory):
        path = os.path.join(directory, "Sessions")
        for i in range(1, 129):
            new_path = os.path.join(path, str(i))
            transcriptions = os.path.join(new_path, 'transcriptions.txt')
            arousal_files = glob.glob(os.path.join(new_path, '*DA.txt'))
            valence_files = glob.glob(os.path.join(new_path, '*DV.txt'))

            #read labels and create line ,label file

    def create_transcriptions(self, directory):
        path = os.path.join(directory, "Sessions")
        for i in range(1, 129):
            new_path = os.path.join(path, str(i))
            transcript = glob.glob(os.path.join(new_path,
                                                 'alignedTranscript_*.txt'))
            word_level_user = glob.glob(os.path.join(new_path,
                                                     'wordLevel*user'))
            word_level_operator = glob.glob(os.path.join(
                new_path, 'wordLevel*operator'))

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
                counter_user_lines = 1
                counter_operator_lines = 1

                for line in t_lines:
                    if line.split("\t")[0].split(" ")[2] == "User":
                        if counter_user_lines < len(user_lines):
                            if user_lines[counter_user_lines][0] == ".":
                                counter_user_lines += 2
                            else:
                                time_start = user_lines[
                                    counter_user_lines].split(" ")[0]

                                while counter_user_lines<len(user_lines) and not (\
                                        user_lines[counter_user_lines][0] == "-"):

                                    if user_lines[counter_user_lines][0] == ".":
                                        pass
                                    else:
                                        time_end = user_lines[
                                            counter_user_lines].split(" ")[1]
                                    counter_user_lines += 1
                                counter_user_lines+=1

                    else:
                        if counter_operator_lines < len(operator_lines):
                            if operator_lines[counter_operator_lines][0] == ".":
                                counter_operator_lines += 2
                            else:
                                time_start = operator_lines[
                                    counter_operator_lines].split(" ")[0]

                                while counter_operator_lines<len(
                                    operator_lines) and (not operator_lines[counter_operator_lines][0] \
                                          == "-"):
                                    if operator_lines[counter_operator_lines][0] == ".":
                                        pass
                                    else:
                                        time_end = operator_lines[
                                            counter_operator_lines].split(" ")[1]
                                    counter_operator_lines += 1

                                counter_operator_lines += 1

                    trans.write((time_start +"\t"+time_end+"\t" +line.split(
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

if __name__=='__main__':
    SemaineDataset("./data/semaine-database_download_2020-01-21_11_41_49")