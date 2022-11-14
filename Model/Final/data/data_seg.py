import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

def standardize_all_data(all_feats):#song_num, seq_len, feat_dim
    X = all_feats
    for i in range(all_feats.shape[2]):
        X[:,:,i] = (X[:,:,i] - np.mean(X[:,:,i]))/np.std(X[:,:,i])
        X[:,:,i] = (X[:,:,i] - np.amin(X[:,:,i]))/(np.amax(X[:,:,i])-np.amin(X[:,:,i]))
    return X

if __name__ == '__main__':
    trs = pd.read_csv('/home/k/kzheng3/Final/data/pmemo_train.csv')
    vs = pd.read_csv('/home/k/kzheng3/Final/data/pmemo_valid.csv')
    ts = pd.read_csv('/home/k/kzheng3/Final/data/pmemo_test.csv')
    total_id = pd.concat((trs,vs,ts),axis = 0)['id']

    segment_feature = pd.read_csv('/home/k/kzheng3/Final/data/spotify/PMEmo_audio_segments.csv')
    seg_feat_id = ['loudness_start','loudness_max',
    'pitches_0','pitches_1','pitches_2','pitches_3','pitches_4','pitches_5','pitches_6','pitches_7','pitches_8','pitches_9','pitches_10','pitches_11',
    'timbre_0','timbre_1','timbre_2','timbre_3','timbre_4','timbre_5','timbre_6','timbre_7','timbre_8','timbre_9','timbre_10','timbre_11']
    min_len = 133#150
    all_feat = []
    count = 0 
    for sid in total_id:
        print(sid)
        seg_features = segment_feature[segment_feature.musicId==sid][seg_feat_id]
        all_feat.append(np.array(seg_features)[:133])
        #break
    print(min_len)
    print(count)
    all_feats = np.array(all_feat)
    print(all_feats.shape)
    np.save('./spotify/all_seg_feats.npy',all_feats)
    # all_feats = np.array(all_feats)
    # np.save('./data/combined_features.npy',all_feats)

    all_feats_normalized = standardize_all_data(all_feats)
    print(np.amax(all_feats_normalized))
   # print(all_feats_normalized)
    np.save('./spotify/seg_features_normalized.npy',all_feats_normalized)
    # print(all_feats.shape)
        