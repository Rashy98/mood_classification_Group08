import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import pickle

def standardize_all_data(all_x,all_ids):# ids order: train, valid, test
    x =[]
    count = 0
    for j in range(3):#train valid test
        ids = all_ids[j]
        for i in ids:
            x.append(all_x[i])
    x = np.array(x)
    print(x.shape)
    x = normalize_matrix(x,len(all_ids[0]))
    return x

def normalize_matrix(x,train_len):#shape num, time, feat_dim 
    train_data = x[:train_len]
    for i in range(x.shape[2]):
        x[:,:,i] = (x[:,:,i]-np.mean(train_data[:,:,i]))/np.std(train_data[:,:,i])
       # x[:,:,i] = (x[:,:,i]-np.amin(train_data[:,:,i]))/(np.amax(train_data[:,:,i]) - np.amin(train_data[:,:,i]))
    return x

if __name__ == '__main__':
    with open('./features_dict.pkl', 'rb') as f:
        all_x = pickle.load(f)
        
    meta_test = pd.read_csv('/home/k/kzheng3/Final/data/pmemo_test.csv')
    meta_train = pd.read_csv('/home/k/kzheng3/Final/data/pmemo_train.csv')
    meta_valid = pd.read_csv('/home/k/kzheng3/Final/data/pmemo_valid.csv')

    target_cols = ['id']

    all_ids = [meta_train.id,meta_valid.id,meta_test.id]
    all_X = standardize_all_data(all_x,all_ids)
    print(all_X.shape,123)
    np.save('./standardized_lyrics.npy',all_X)
    #print(all_X[0,0])
   # np.save('./min_max_lyrics.npy',all_X)
   
        