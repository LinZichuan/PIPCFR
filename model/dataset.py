import torch
import pandas as pd


class ESXDataset(torch.utils.data.Dataset):
    def __init__(self, train_X, train_next_X, train_Y, train_T):
        self.train_X = torch.from_numpy(train_X).float()
        self.train_next_X = torch.from_numpy(train_next_X).float()
        self.train_T = torch.from_numpy(train_T).float()  # label of treatment status
        self.train_Y = torch.from_numpy(train_Y).float()  # label of conversion
        self.data_num = len(train_X)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_x = self.train_X[idx]
        out_next_x = self.train_next_X[idx]
        out_t = self.train_T[idx]
        out_y = self.train_Y[idx]
        return out_x, out_next_x, out_t, out_y
