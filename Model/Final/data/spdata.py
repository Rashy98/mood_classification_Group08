import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

def standardize_all_data():
    print(123)

if __name__ == '__main__':
    trs = pd.read_csv('/home/k/kzheng3/Final/data/pmemo_train.csv')
    vs = pd.read_csv('/home/k/kzheng3/Final/data/pmemo_valid.csv')
    ts = pd.read_csv('/home/k/kzheng3/Final/data/pmemo_test.csv')
    total_id = pd.concat((trs,vs,ts),axis = 0)['id']
    #total_id = total['id'].to_numpy()

    spotify_feature = pd.read_csv('/home/k/kzheng3/Final/data/spotify/PMEmo_audio_features.csv')
    segment_feature = pd.read_csv('/home/k/kzheng3/Final/data/spotify/PMEmo_audio_segments_cleaned.csv')
    all_feats = []
    song_feat_id = ['key','mode','energy','danceability','loudness','tempo']
    seg_feat_id = ['loudness_start','loudness_max','loudness_end',
    'pitches_0','pitches_1','pitches_2','pitches_3','pitches_4','pitches_5','pitches_6','pitches_7','pitches_8','pitches_9','pitches_10','pitches_11',
    'timbre_0','timbre_1','timbre_2','timbre_3','timbre_4','timbre_5','timbre_6','timbre_7','timbre_8','timbre_9','timbre_10','timbre_11']
    for sid in total_id:
        print(sid)
        song_features = np.array(spotify_feature[spotify_feature.musicId==sid][song_feat_id])
        seg_features = segment_feature[segment_feature.musicId==sid][seg_feat_id]
        avg_seg_features = np.mean(np.array(seg_features),axis=0)
        combine_feat = np.append(song_features,avg_seg_features)
        all_feats.append(combine_feat)
    all_feats = np.array(all_feats)
    np.save('./spotify/combined_features.npy',all_feats)
    scaler = MinMaxScaler()
    scaler.fit(all_feats)
    all_feats_normalized = scaler.transform(all_feats)
    np.save('./spotify/combined_features_normalized.npy',all_feats_normalized)
    print(all_feats.shape)
        