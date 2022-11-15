import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import weight_norm
from torch.autograd import Variable

import numpy as np

"""
Written by: Kexin Zheng
"""

class RNN_encoder(nn.Module):
    def __init__(self, input_size = 26, hidden_size = 64, num_layers = 2):
        super(RNN_encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)

    
    def forward(self, x):
        out, _ = self.lstm(x) 
        out = self.dropout(out)
        return out

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

class WeightNet(nn.Module):
    def __init__(self, dims):
        super(WeightNet, self).__init__()
        self.num_layers = len(dims) -1
        self.drop = 0.2
        self.seq = nn.Sequential(*self._init_layers(dims))
 
    def _init_layers(self, dims):
        layers = []
        for i in range(self.num_layers):
            layers.append(weight_norm(nn.Linear(dims[i], dims[i + 1]), dim=None))
            layers.append(nn.ReLU())
        return layers

    def forward(self, x):
        return self.seq(x)

class CrossAtt(nn.Module):
    def __init__(self, v_dim, q_dim, hid_dim, dropout=0.2):
        super(CrossAtt, self).__init__()
        self.v_proj = WeightNet([v_dim, hid_dim])
        self.q_proj = WeightNet([q_dim, hid_dim])
        self.drop = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(hid_dim, 1), dim=None)

    def forward(self, v, q):
        #v: [b, k, v_dim]
        #q: [b, q__dim]
        v_proj = self.v_proj(v)  # [b, k, v_dim]
        q_proj = self.q_proj(q).unsqueeze(1) # [b, 1, q_dim]
        logits = self.linear(self.drop(v_proj * q_proj))
        out = nn.functional.softmax(logits, 1)
        return out

class fuse_att(nn.Module):#input_shape= (224,224,3)
    def __init__(self,  label_num = 2):
        super(fuse_att, self).__init__() 
        self.spect_model = CNN()
        self.sptf_model = RNN_encoder()

        self.proj1 = nn.Linear(14*14*64, 128)
        self.proj2 = nn.Linear(133*64, 128)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)
        self.norm = nn.LayerNorm(128)
        self.relu = nn.ReLU(inplace=True)

        self.attention1 = CrossAtt(v_dim=64,q_dim =128,hid_dim=256)
        self.attention2 = CrossAtt(v_dim=64,q_dim =128,hid_dim=256)


    def forward(self, spect,sptf):
        spect_fet = self.spect_model(spect) # batch, 64, 14 * 14
        spect_fet = spect_fet.view(spect_fet.size(0), spect_fet.size(1), -1)

        sptf_fet = self.sptf_model(sptf) # b, 133, 128

        proj_feat_spect = self.proj1(spect_fet.view(spect_fet.size(0),-1)) # b,128
        proj_feat_sptf = self.proj2(sptf_fet.contiguous().view(sptf_fet.size(0),-1))# b,128

        weight1 = self.attention1(sptf_fet, proj_feat_spect) # 1, 133, 1
        feat_att_sptf = torch.squeeze(torch.matmul(weight1.permute(0,2,1), sptf_fet),dim=1)
        weight2 = self.attention2(spect_fet.permute(0,2,1), proj_feat_sptf)
        feat_att_spect = torch.squeeze(torch.matmul(weight2.permute(0,2,1), spect_fet.permute(0,2,1)),dim=1)
        feat_cat = torch.cat((feat_att_sptf, feat_att_spect), dim=1)

        feat_cat = self.norm(feat_cat)
        out = self.relu(self.fc1(feat_cat))
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    model = fuse_att()
    spects = torch.rand(size=(3, 3, 224, 224))
    spty = torch.rand((3,133,26))
    logits = model(spects,spty) # # 1,64,14,14
    print(logits.shape)
