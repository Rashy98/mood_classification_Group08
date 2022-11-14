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



class RNN_encoder(nn.Module):
    def __init__(self, input_size = 26, hidden_size = 128, num_layers = 2):
        super(RNN_encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)

       # self.fc1= nn.Linear(hidden_size, hidden_size)
        self.fc2= nn.Linear(hidden_size, num_layers)

    
    def forward(self, x):
        out, _ = self.lstm(x) 
        out = self.dropout(out)
        out = torch.mean(out,dim = 1)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    model = CNN()
    # print(model.spect)
    spect = torch.rand((1,3,224,224))
    logits = model(spect)
    print(logits.shape)
    model.load_state_dict(torch.load('./valence.pth'))
    # with torch.no_grad():
    #     print(model.conv1.weight.data)

#    # res = model(x)

# model = Sequential()
#model.add(Conv2D(8, (3,3), activation= 'relu', input_shape= (224,224,3), padding= 'same'))
# model.add(MaxPooling2D((4,4), padding= 'same'))
# model.add(Conv2D(16, (3,3), activation= 'relu', padding= 'same'))
# model.add(MaxPooling2D((4,4), padding= 'same'))
# model.add(Conv2D(32, (3,3), activation= 'relu', padding= 'same'))
# model.add(MaxPooling2D((4,4), padding= 'same'))
# model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
# model.add(MaxPooling2D((4,4), padding= 'same'))
# model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
# model.add(MaxPooling2D((4,4), padding= 'same'))
# model.add(Flatten())
# model.add(Dense(256, activation= 'relu'))
# model.add(Dense(4, activation= 'softmax'))