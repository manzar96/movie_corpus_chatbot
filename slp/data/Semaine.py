from torch.utils.data import Dataset
import glob
import os


class SemaineDataset(Dataset):
    def __init__(self, directory, transforms=None, train=True):

        self.read_data(directory)


    def read_data(self, directory):
        path = os.path.join(directory, "Sessions")
        session_files = []
        for i in range(1, 3):
            new_path = os.path.join(path, str(i))
            file_list = glob.glob(os.path.join(new_path, "*.txt"))
            session_files.append(file_list)
        for session in session_files:
            transcripts = glob.glob('aligned_Transcript_*.txt')
            word_level_user = glob.glob('wordLevel_*_user')
            word_level_operator = glob.glob('wordLevel_*_operator')

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