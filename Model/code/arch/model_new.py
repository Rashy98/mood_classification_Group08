import torch
import torch.nn as nn
from torch.autograd import Variable

class MultiNet(nn.Module):
    def __init__(self, emb_dim = 100, hid_dim = 40):
        super().__init__()
        self.num_output = 2
        self._lstm_hid_dim = hid_dim
        self._conv0 = nn.Sequential(nn.Conv1d(in_channels=emb_dim, out_channels=16, kernel_size=2, stride=2),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2))
        self._lstm = nn.LSTM(16, hid_dim, batch_first=False)
        self._classifier = nn.Sequential(nn.Linear(in_features=480, out_features=64),
                                         nn.Tanh(),
                                         nn.Dropout(0.5),
                                         nn.Linear(in_features=64, out_features=self.num_output))
        self.apply(self._initalize)

    def forward(self, x):
        x = x.permute(0, 2, 1)      # ([32, 100, 50]) Batch, Input_D, seq_len
        x = self._conv0(x)        # ([32, 16, 25]) Batch, Input_D, seq_len
        x = x.permute(2, 0, 1)        # ([25, 32, 16]) seq_len, Batch, Input_D
        lstm_out, (h,c) = self._lstm(x)
        lstm_out = lstm_out.permute(1,2,0)    # ([32, 40, 25])   # Batch, Input_D, seq_len
        x = lstm_out.contiguous()
        x = x.view(x.size(0), -1)
        x = self._classifier(x) # maybe remove this layer for fusion model
        return x

    def _initalize(self, layer):
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_uniform_(layer.weight)

if __name__ == '__main__':
    x = torch.rand((32,50,100))
    model = MultiNet()
    logits = model(x)
    print(logits.shape)
