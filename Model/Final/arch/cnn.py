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



class CNN(nn.Module):#input_shape= (224,224,3)
    def __init__(self, input_dim = 3, label_num = 2):
        super(CNN, self).__init__()  
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=8, kernel_size=(3, 3), padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
       # self.bn1 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same')
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))
       # self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(14*14*64,label_num)
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.pool1(out)
        out = self.relu(self.conv2(out))
        out = self.pool2(out)
        out = self.dropout(out)
       # out = self.bn1(out)

        out = self.relu(self.conv3(out))
        out = self.pool3(out)
        out = self.relu(self.conv4(out))
        out = self.pool4(out)
        out = self.dropout(out)
       # out = self.bn2(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
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
    model_trained = load_model('/home/k/kzheng3/Final/arch/arousal_model_3.h5')
    extracted_weights = model_trained.layers[0].get_weights()
    # print(len(kweight))
    with torch.no_grad():
        kweight = np.transpose(model_trained.layers[0].get_weights()[0], (3, 2, 0, 1))
        model.conv1.weight.data = torch.tensor(kweight,dtype = torch.float32)
        print(kweight)
        kweight = np.transpose(model_trained.layers[2].get_weights()[0], (3, 2, 0, 1))
        model.conv2.weight.data = torch.tensor(kweight,dtype = torch.float32)
        kweight = np.transpose(model_trained.layers[4].get_weights()[0], (3, 2, 0, 1))
        model.conv3.weight.data = torch.tensor(kweight,dtype = torch.float32)
        kweight = np.transpose(model_trained.layers[6].get_weights()[0], (3, 2, 0, 1))
        model.conv4.weight.data = torch.tensor(kweight,dtype = torch.float32)
        torch.save(model.state_dict(), './arousal.pth')
        

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