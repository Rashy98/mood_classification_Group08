import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

import numpy as np



class CNN(nn.Module):#input_shape= (224,224,3)
    def __init__(self, input_dim = 3, label_num = 2):
        super(CNN, self).__init__()  
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=8, kernel_size=(3, 3), padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same')
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.relu = nn.ReLU(inplace = True)
        self.dropout =  torch.nn.Dropout(0.2)


    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.pool1(out)
        out = self.relu(self.conv2(out))
        out = self.pool2(out)
        out = self.dropout(out)

        out = self.relu(self.conv3(out))
        out = self.pool3(out)
        out = self.relu(self.conv4(out))
        out = self.pool4(out)

        return out

# Lyric encoder by Dane
class MultiNetEncoder(nn.Module):
    def __init__(self, emb_dim = 100, hid_dim = 40):
        super().__init__()
        self.num_output = 2
        self._lstm_hid_dim = hid_dim
        self._conv0 = nn.Sequential(nn.Conv1d(in_channels=emb_dim, out_channels=16, kernel_size=2, stride=2),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2))
        self._lstm = nn.LSTM(16, hid_dim, batch_first=False)
        self.apply(self._initalize)

    def forward(self, x):
        x = x.permute(0, 2, 1)      # ([32, 100, 50]) Batch, Input_D, seq_len
        x = self._conv0(x)        # ([32, 16, 25]) Batch, Input_D, seq_len
        x = x.permute(2, 0, 1)        # ([25, 32, 16]) seq_len, Batch, Input_D
        lstm_out, (h,c) = self._lstm(x)
        lstm_out = lstm_out.permute(1,2,0)    # ([32, 40, 25])   # Batch, Input_D, seq_len
        x = lstm_out.contiguous()
        x = x.view(x.size(0), -1)
        return x

    def _initalize(self, layer):
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_uniform_(layer.weight)

class fuse_ls(nn.Module):#input_shape= (224,224,3)
    def __init__(self, input_dim = 3, label_num = 2):
        super(fuse_ls, self).__init__() 
        self.spect_model = CNN()
        self.lyric_model = MultiNetEncoder()

        self.proj1 = nn.Linear(14*14*64, 128)
        self.proj2 = nn.Linear(480, 128)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)
        self.norm = nn.LayerNorm(256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, spect,lyric):
        fet_spect = self.spect_model(spect)
        fet_lyric = self.lyric_model(lyric)
        spect_feat = self.proj1(fet_spect.view(fet_spect.size(0),-1))
        lyric_feat = self.proj2(fet_lyric)

        feat_cat = torch.cat((spect_feat, lyric_feat), dim=1)

        feat_cat = self.norm(feat_cat)
        out = self.relu(self.fc1(feat_cat))
        out = self.fc2(out)

        return out

if __name__ == '__main__':
    model = fuse_ls()
    spects = torch.rand(size=(1, 3, 224, 224))
    lyric = torch.ones(size=(1,50,100))
    logits = model(spects,lyric) # # 1,64,14,14

    print(logits.shape)
