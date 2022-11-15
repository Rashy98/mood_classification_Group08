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


class fuse_cs_direct(nn.Module):#input_shape= (224,224,3)
    def __init__(self,label_num = 2):
        super(fuse_cs_direct, self).__init__() 
        self.spect_model = CNN()
            
        self.sptf_fc = nn.Sequential(
            nn.Linear(33,256),
            nn.ReLU(inplace = True),
        )

        self.proj1 = nn.Linear(14*14*64, 128)
        self.proj2 = nn.Linear(256, 128)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)
        self.norm = nn.LayerNorm(256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, spect,sptf):
        spect_fet = self.spect_model(spect)
        sptf_fet = self.sptf_fc(sptf)

        feat_spect = self.proj1(spect_fet.view(spect_fet.size(0),-1))
        feat_sptf = self.proj2(sptf_fet)
        feat_cat = torch.cat((feat_spect, feat_sptf), dim=1)

        out = self.relu(self.fc1(feat_cat))
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    model = fuse_cs_direct()
    spects = torch.rand((1,3,224,224))
    sptf = torch.ones(1,33)
    model(spects,sptf)
