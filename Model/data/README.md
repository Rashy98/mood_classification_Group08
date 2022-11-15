# Datasets
This folder contains code and metadata for different datasets.<br>
# PMEmo Dataset
For original dataset please refer to :<a href=https://github.com/HuiZhangDB/PMEmo>PMEmo</a><br>
**PMEmo_spotify_feature.py** contains code for retrieve global music feature using spotify API.<br>
**PMEmo_dynamic_features.py** contains code for retrieve segment-level music feature using spotify API.<br>
**PMEmo_split.py** contains code for splits PMEmo into train, test, and validation set.<br>
**PMEmo_relabel.py** contains code to relabel the numerical emotion feature into classification label.<br>
# NJU Dataset
For original dataset please refer to :<a href=https://cs.nju.edu.cn/sufeng/data/musicmood.htm>NJU Mood</a><br>
**NJU_spotify.ipynb** contains code for retrieve spotify features for NJU dataset and dataset splitting
# Medieval dataset
For original datast please refer to <a href=https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music>Deam</a><br>
**MediaEval_clean.ipynb** contains code for spliting medieval dataset<br>
**resample.py** contains code for resampling audio data to 16kHz to match wav2vec configuration<br>
To further match the wav2vec setting, please use <a href=https://github.com/facebookresearch/demucs>Demucus</a> to separate human vocal from background music beforehand.<br>
**local_feature.py** contains code for extracting local feature using wav2vec2.0 model<br>
**context_feature.py.py** contains code for extracting contextualized feature using wav2vec2.0 model<br>
