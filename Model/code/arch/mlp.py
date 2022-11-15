import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import numpy as np



class MLP(nn.Module):
    def __init__(self, input_dim = 26, label_num = 2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,label_num)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
   model = MLP(input_dim = 33)
   sptf = torch.rand((1,33))
   logits = model(sptf)
   