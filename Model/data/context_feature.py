import fairseq
import soundfile as sf

import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np

class wavDataset(Dataset):
    def __init__(self,file_path):
        self.x_train = []
        self.x_name = []
        vocal_list = os.listdir(file_path)
        max_length = 240000#320000#288000
        for vocal_file in vocal_list:
            if vocal_file.endswith('.mp3'):
                x,s = sf.read(os.path.join(file_path,vocal_file))
                max_pos = np.argmax(x)
                if max_pos <max_length/2:
                    x = x[:max_length]
                elif max_pos > len(x) - max_length:
                    x = x[-max_length:]
                else:
                    x = x[max_pos - int(max_length/2):max_pos + int(max_length/2)]
                x = x[None, :]
                self.x_train.append(torch.tensor(x,dtype=torch.float32))
                self.x_name.append(vocal_file.split('.')[0])

    def __len__(self):
        return len(self.x_name)
    
    def __getitem__(self,idx):
        return self.x_train[idx],self.x_name[idx]

if __name__ == '__main__':
    # file_path = '/home/k/kzheng3/MER/medieval/vocal_resampled'
    # save_path = '/home/k/kzheng3/MER/medieval/w2v_context_large'
    file_path = '/home/k/kzheng3/MER/vocal/resampled'
    save_path = '/home/k/kzheng3/MER/pmemow2v/context'
    device = 'cuda'
    wavset = wavDataset(file_path)

   # cp = './libri960_big.pt'
    cp = '/home/k/kzheng3/MER/wav2vec/wav2vec_small.pt'
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp])
    model = model[0].to(device)
    model.extractor_mode='layer_norm'
    model.eval()
            
    for step, (data,label) in enumerate(tqdm(wavset)):
        data = data.to(device)
        z = model.extract_features(data,padding_mask = None, mask=False, layer=None)['x']
        print(os.path.join(save_path,label+'.npy'))
        np.save(os.path.join(save_path,label+'.npy'),z.detach().cpu().numpy())
