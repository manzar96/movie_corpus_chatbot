from torch.utils.data import Dataset
import glob
import os
import re

class SemaineDataset(Dataset):
    def __init__(self, directory, transforms=None, train=True):

        self.read_data(directory)

    def read_data(self, directory):
        path = os.path.join(directory, "Sessions")
        counter = 0
        for i in range(1,3):
            new_path = os.path.join(path, str(i))
            transcript = glob.glob(os.path.join(new_path,
                                                 'alignedTranscript_*.txt'))
            word_level_user = glob.glob(os.path.join(new_path,
                                                     'wordLevel*user'))
            word_level_operator = glob.glob(os.path.join(
                new_path, 'wordLevel*operator'))

            if not transcript == []:
                with open(transcript[0],"r")as tfile:
                    lines = tfile.readlines()
                    for line in lines:
                        start_time = line.split("\t")[0].split(" ")[0]
                        end_time = line.split("\t")[0].split(" ")[1]
                        utt = line.split("\t")[1]
                        utt = re.sub("\([^)]+\).","",utt)
                        utt = utt.strip()
                        print(start_time)
                        print(end_time)
                        print(utt)

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