import numpy as np
import os
import pickle
from torch.utils.data import DataLoader, SubsetRandomSampler


def dataloaders_from_indices(dataset, train_indices, val_indices, batch_train,
                             batch_val, collator_fn):
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_train,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=collator_fn)
    val_loader = DataLoader(
        dataset,
        batch_size=batch_val,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=collator_fn)
    return train_loader, val_loader


def train_test_split(dataset, batch_train, batch_val, collator_fn,
                     test_size=0.2, shuffle=True, seed=None):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_size * dataset_size))
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices = indices[test_split:]
    val_indices = indices[:test_split]
    return dataloaders_from_indices(dataset, train_indices, val_indices,
                                    batch_train, batch_val, collator_fn)

def make_train_val_test_pickles(outfolder,dataset,val_size,test_size,\
                                         shuffle=True,
                                seed=None):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_test_split = int(np.floor((test_size+val_size) * dataset_size))
    test_split = int(np.floor(test_size *dataset_size))
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
    train_indexes = indices[val_test_split:]
    test_indexes = indices[:test_split]
    val_indexes = indices[test_split:val_test_split]
    train_dataset = list(map(dataset.__getitem__, train_indexes))
    val_dataset = list(map(dataset.__getitem__, val_indexes))
    test_dataset = list(map(dataset.__getitem__, test_indexes))
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    with open(os.path.join(outfolder,'train_set.pkl'),'wb') as handle:
        pickle.dump(train_dataset,handle,protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    with open(os.path.join(outfolder, 'val_set.pkl'), 'wb') as handle:
        pickle.dump(val_dataset,handle,protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    with open(os.path.join(outfolder, 'test_set.pkl'), 'wb') as handle:
        pickle.dump(test_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
