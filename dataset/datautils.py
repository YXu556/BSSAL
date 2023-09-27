import random
import numpy as np

import torch
from torch.utils.data import Dataset


# ---------------------------- Spectral augmentation ----------------------------
class RandomChanSwapping:
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x.reshape(-1, 10)
            s_idx = random.sample(range(x.shape[1]), 2)
            idx = list(range(x.shape[1]))
            idx[s_idx[0]] = s_idx[1]
            idx[s_idx[1]] = s_idx[0]
            x = x[:, idx]
            x = x.reshape(-1)
        return x


class RandomChanRemoval:
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x.reshape(-1, 10)
            s_idx = random.sample(range(x.shape[1]), 2)
            idx = list(range(x.shape[1]))
            idx[s_idx[0]] = s_idx[1]
            x = x[:, idx].reshape(-1)
        return x


class RandomAddNoise:
    def __call__(self, x):
        x = x.reshape(-1, 10)
        t, c = x.shape
        for i in range(t):
            prob = np.random.rand()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.5:
                    x[i, :] += -np.abs(np.random.randn(c) * 0.2)
                else:
                    x[i, :] += np.abs(np.random.randn(c) * 0.2)
        x = x.reshape(-1)
        return x


# ---------------------------- Temporal augmentation ----------------------------
class RandomTempSwapping:
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x.reshape(-1, 10)
            s_idx = random.sample(range(x.shape[0]), 2)
            idx = list(range(x.shape[0]))
            idx[s_idx[0]] = s_idx[1]
            idx[s_idx[1]] = s_idx[0]
            x = x[idx].reshape(-1)
        return x


class RandomTempShift:
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x.reshape(-1, 10)
            shift = int(np.clip(np.random.randn() * 0.3, -1, 1) * 3)
            x = np.roll(x, shift, axis=0)
            x = x.reshape(-1)

        return x


class ToTensor:
    """Transform the image to tensor.
    """

    def __init__(self, in_ch):
        self.in_ch = in_ch

    def __call__(self, x):
        x = torch.from_numpy(x[:self.in_ch])
        return x
