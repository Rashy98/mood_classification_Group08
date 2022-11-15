# Model training
This folder contains codes for evaluating different modalities on the PMEmo dataset.

## Codes
**train_cnn.py** contains codes for training CNN spectrogram<br>
**train_fuse_attls.py** contains codes for training cross-attention feature fusion for spectrogram and segment features<br>
**train_fuse.py** contains codes for training contatenation feature fusion for spectrogram and global features<br>
**train_fuse_cnnls.py** contains codes for contatenation feature fusion for spectrogram and segment features<br>
**train_fuse_lyric.py** contains codes for training contatenation feature fusion for spectrogram and lyric features<br>
**train_lyric.py**contains code for training CNN+LSTM encoder for lyric features<br>
**train_rnn.py** contains code for training LSTM encoder for segment features<br>
**train_mlp.py** contains code for training MLP for global features<br>
<br>
**dataset.py** contains code for loading dataset for training

## How to run
To predict valence
```bash
python train_fuse.py --emo_dim valence
```
To predict arousal
```bash
python train_fuse.py --emo_dim arousal
```
the **train_fuse.py** can be replaced by any of the aforementioned training codes.
## Data
The data folder contains processed data files for training
## Pretrained
The pretrained folder contains model pretrained on other dataset which will be used for further fine-tuning on the PMEmo dataset
## Arch
The arch folder contains model structures.<br>
**cnn.py** contains codes for CNN encoder<br>
**fuse_attlstm.py** contains codes for cross-attention feature fusion for spectrogram and segment features<br>
**fuse_cnnmlp.py** contains codes for contatenation feature fusion for spectrogram and global features<br>
**fuse_cnnlstm.py** contains codes for contatenation feature fusion for spectrogram and segment features<br>
**fuse_ls.py** contains codes for contatenation feature fusion for spectrogram and lyric features<br>
**model_new.py**contains code for CNN+LSTM encoder for lyric features<br>
**rnn.py** contains code for LSTM encoder for segment features<br>
**mlp.py** contains code for MLP encoder for global features

## Screenshot
The screenshot of example runs are shown below
![result screenshot 1](https://github.com/Rashy98/mood_classification_Group08/blob/master/Model/code/results/Screenshot%202022-11-10%20at%204.15.33%20PM.png)
![result screenshot 2](https://github.com/Rashy98/mood_classification_Group08/blob/master/Model/code/results/Screenshot%202022-11-11%20at%206.18.04%20PM.png)
![result screenshot 2](https://github.com/Rashy98/mood_classification_Group08/blob/master/Model/code/results/Screenshot%202022-11-15%20at%208.16.25%20AM.png)

