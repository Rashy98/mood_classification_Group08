import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
# import keras
# from keras.models import load_model
import numpy as np



class RNN(nn.Module):
    def __init__(self, input_size = 26, hidden_size = 64, num_layers = 2):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc2= nn.Linear(hidden_size, num_layers)

    
    def forward(self, x):
        out, _ = self.lstm(x) 
        out = self.dropout(out)
        out = torch.mean(out,dim = 1)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    model = RNN()
