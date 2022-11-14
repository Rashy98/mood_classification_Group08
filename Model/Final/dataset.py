import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os

class spotifyDataset(Dataset):
    def __init__(self,all_x,meta_set):
        id_col = meta_set.columns[0]
        y_col = meta_set.columns[1]
        self.x = torch.tensor(all_x[meta_set[id_col]],dtype = torch.float32)
        self.y = torch.tensor(meta_set[y_col],dtype = torch.float32)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

class cnnDataset(Dataset):
    def __init__(self,all_x,set_y,start_idx):
        end_idx = start_idx + len(set_y)
        self.x = torch.tensor(all_x[start_idx:end_idx],dtype = torch.float32)
        self.y = torch.tensor(set_y[set_y.columns[1]],dtype = torch.float32)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

class fusecnnlmDataset(Dataset):
    def __init__(self,spect_x,sptf_x,set_y,start_idx):
        end_idx = start_idx + len(set_y)
        self.spect = torch.tensor(spect_x[start_idx:end_idx],dtype = torch.float32)
        self.sptf = torch.tensor(sptf_x[start_idx:end_idx],dtype = torch.float32)
        self.y = torch.tensor(set_y[set_y.columns[1]],dtype = torch.float32)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return (self.spect[idx],self.sptf[idx]), self.y[idx]

class lyricDataset(Dataset):
    def __init__(self,all_x,set_y,start_idx):
        end_idx = start_idx + len(set_y)
        self.x = torch.tensor(all_x[start_idx:end_idx],dtype = torch.float32)
        self.y = torch.tensor(set_y[set_y.columns[1]],dtype = torch.float32)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]



if __name__ == '__main__':
    print(123)




