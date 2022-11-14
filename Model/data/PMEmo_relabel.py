import numpy as np
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth,SpotifyClientCredentials
import tqdm
import pandas as pd
import shutil

def va_to_cat(r,dl):
    d = dl[dl.musicId == r.musicId]
    if d['Valence(mean)'].to_numpy()[0]>0.5: 
        return 'High Valence'
    else:
        return 'Low Valence'

def ar_to_cat(r,dl):
    d =dl[dl.musicId == r.musicId]
    if d['Arousal(mean)'].to_numpy()[0]>0.5: 
        return 'High Arousal'
    else:
        return 'Low Arousal'

if __name__ == '__main__':
    df_train = pd.read_csv('./PMEmo_train.csv')
    df_valid = pd.read_csv('./PMEmo_valid.csv')
    df_test = pd.read_csv('/PMEmo_test.csv')
    df_label = pd.read_csv('./PMEmo/annotations/static_annotations.csv')

    df_train['valence_category'] = df_train.apply(lambda r: va_to_cat(r,df_label),axis = 1)
    df_valid['valence_category'] = df_valid.apply(lambda r: va_to_cat(r,df_label),axis = 1)
    df_test['valence_category'] = df_test.apply(lambda r: va_to_cat(r,df_label),axis = 1)

    df_train['arousal_category'] = df_train.apply(lambda r: ar_to_cat(r,df_label),axis = 1)
    df_valid['arousal_category'] = df_valid.apply(lambda r: ar_to_cat(r,df_label),axis = 1)
    df_test['arousal_category'] = df_test.apply(lambda r: ar_to_cat(r,df_label),axis = 1)


    df_train.to_csv('./data/PMEmo_train.csv',index= False)
    df_valid.to_csv('./data/PMEmo_valid.csv',index= False)
    df_test.to_csv('./data/PMEmo_test.csv',index= False)

