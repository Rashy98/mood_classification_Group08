from email.mime import audio
import numpy as np
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth,SpotifyClientCredentials
import tqdm
import pandas as pd
"""
Retrieve global-level music feature using spotify API
"""
def get_audio_features(spotifyID):
    #"key""mode""acousticness""energy""danceability""instrumentalness""loudness""tempo""valence"
    features = ['key',"mode","acousticness","energy","danceability","instrumentalness","loudness","tempo","valence"]
    audio_info = sp.audio_features(spotifyID)[0]
    audio_features = [spotifyID]
    for f in features:
        audio_features.append(audio_info[f])
    return audio_features


if __name__ == '__main__':
    print("start")
    metadata = pd.read_csv('./PMEmo/metadata.csv')
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="0eee6e2465a94368ad5ec7450a2fd62a",
                                               client_secret="35113bd096574a4dae4381bc4090f7ba"))

    audio_features = []
    for idx, track in metadata.iterrows():
        res = get_audio_features(track.spotifyId)
        res.insert(0,track.musicId)
        audio_features.append(res)
    df = pd.DataFrame(audio_features, columns =['musicId','spotifyId','key',"mode","acousticness","energy","danceability","instrumentalness","loudness","tempo","valence"]) 
    df.to_csv('./data/PMEmo_audio_features.csv')

