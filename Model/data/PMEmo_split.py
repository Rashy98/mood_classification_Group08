import numpy as np
import pandas as pd

# Read metadata
df = pd.read_csv('./PMEmo/metadata.csv')
df_ref = pd.read_csv('./PMEmo/PMEmo_audio_features.csv')
df_Labels = pd.read_csv('./PMEmo/annotations/static_annotations.csv')

mask = np.isin(df.musicId,df_ref.musicId) # data that has spotify features
df_in = df[mask]

# Split metadata (with spotify feature) into train test and valid with a ratio of 8:1:1
indx = np.arange(len(df_in))
np.random.shuffle(indx) # randomize
train_num = int(len(df_in) * 0.8)
test_num = int(len(df_in) * 0.1)
dt = df_in.iloc[indx[:train_num]]
dtt = df_in.iloc[indx[train_num:train_num+test_num]]
dv= df_in.iloc[indx[train_num+test_num:]]

dt = dt.sort_values(by='musicId')
dv = dv.sort_values(by='musicId')
dtt = dtt.sort_values(by='musicId')

dt.to_csv('./data/metadata_train.csv',index = False)
dv.to_csv('./data/metadata_valid.csv',index = False)
dtt.to_csv('./data/metadata_test.csv',index = False)
