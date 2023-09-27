import random
import numpy as np
from collections import Counter

import torch
from torch.utils.data import Dataset


class WrapDataset(Dataset):

    def __init__(self, x, y, transform=None, site=None, ndvi=False):
        self.x = x
        self.y = y - 1

        self.mean = 0.2
        self.std = 0.1

        if ndvi:
            x_ts = x.reshape(x.shape[0], -1, 10)
            self.ndvi = np.clip((x_ts[:, :, 6] - x_ts[:, :, 2]) / (x_ts[:, :, 6] + x_ts[:, :, 2]+1e-6), -1, 1)
        # normalize
        self.x = self.x / 10000
        self.x = (self.x - self.mean) / self.std

        self.transform = transform
        self.site = site

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        data, target = self.x[idx], self.y[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, target


class ALDataset():
    def __init__(self, x, y, num_val, select_type, site=None):
        self.x_pool = x
        self.y_pool = y
        self.select_type = select_type
        self.site = site

        num_samples = self.x_pool.shape[0]
        indices = np.arange(num_samples)

        self.val_idx_list = np.random.choice(indices, num_val, replace=False)
        print(Counter(y[self.val_idx_list]))
        self.labeled_idx_list = []
        self.unlabeled_idx_list = np.delete(indices, self.val_idx_list)

    def get_init(self, num):
        self.labeled_idx_list = np.random.choice(self.unlabeled_idx_list, num, replace=False).tolist()
        self.unlabeled_idx_list = np.setdiff1d(self.unlabeled_idx_list, self.labeled_idx_list)

    def update(self, step, *args):
        if self.select_type == 1:
            selected = np.random.choice(len(self.unlabeled_idx_list), step, replace=False)
            self.labeled_idx_list += self.unlabeled_idx_list[selected].tolist()
            self.unlabeled_idx_list = np.delete(self.unlabeled_idx_list, selected)
        else:
            pooled_index = args[0]
            self.labeled_idx_list += self.unlabeled_idx_list[pooled_index].tolist()
            self.unlabeled_idx_list = np.delete(self.unlabeled_idx_list, pooled_index)

    def get_samples(self):
        x_labeled = self.x_pool[self.labeled_idx_list]
        y_labeled = self.y_pool[self.labeled_idx_list]
        x_valid = self.x_pool[self.val_idx_list]
        y_valid = self.y_pool[self.val_idx_list]
        x_unlabeled = self.x_pool[self.unlabeled_idx_list]
        y_unlabeled = self.y_pool[self.unlabeled_idx_list]

        samples = {'x_labeled': x_labeled,
                   'y_labeled': y_labeled,
                   'x_valid': x_valid,
                   'y_valid': y_valid,
                   'x_unlabeled': x_unlabeled,
                   'y_unlabeled': y_unlabeled}

        if self.site is not None:
            site_labeled = self.site[self.labeled_idx_list]
            site_unlabeled = self.site[self.unlabeled_idx_list]
            samples.update({'site_labeled': site_labeled, 'site_unlabeled': site_unlabeled})
        return samples

    def get_labeled_y(self):
        return self.y_pool[self.labeled_idx_list]

    def get_labeled_site(self):
        return self.site[self.labeled_idx_list]

    def get_labeled_idx(self):
        return self.labeled_idx_list