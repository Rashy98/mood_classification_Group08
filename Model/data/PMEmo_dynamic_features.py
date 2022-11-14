from email.mime import audio
import numpy as np
import pandas as pd

import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth,SpotifyClientCredentials
"""
Retrieve segment-level music feature using spotify API
"""
def get_audio_analysis(spotifyID):
    audio_info = sp.audio_analysis(spotifyID)
    audio_segs = audio_info['segments']
    audio_secs = audio_info['sections']
    audio_trackinfo = audio_info['track']#"analysis_sample_rate" "analysis_channels""time_signature""key""mode"
    useful_info = []
    useful_info.append(spotifyID)
    useful_info.append(audio_trackinfo['analysis_sample_rate'])
    useful_info.append(audio_trackinfo['analysis_channels'])
    return segment_info_list(audio_segs,useful_info),section_info_list(audio_secs,useful_info)

def segment_info_list(segs,track_info):
    segs_list = []
    for seg in segs:
        seg_list = [track_info[0],track_info[1],track_info[2]]
        seg_list.append(seg["start"])
        seg_list.append(seg["duration"])
        seg_list.append(seg["confidence"])
        seg_list.append(seg["loudness_start"])
        seg_list.append(seg["loudness_max"])
        seg_list.append(seg["loudness_max_time"])
        seg_list.append(seg["loudness_end"])
        seg_list.extend(seg["pitches"])
        seg_list.extend(seg["timbre"])
        segs_list.append(seg_list)
    return segs_list

def section_info_list(segs,track_info):
    segs_list = []
    for seg in segs:
        seg_list = [track_info[0],track_info[1],track_info[2]]
        seg_list.append(seg["start"])
        seg_list.append(seg["duration"])
        seg_list.append(seg["confidence"])
        seg_list.append(seg["loudness"])
        seg_list.append(seg["tempo"])
        seg_list.append(seg["tempo_confidence"])
        seg_list.append(seg["key"])
        seg_list.append(seg["key_confidence"])
        seg_list.append(seg["mode"])
        seg_list.append(seg["mode_confidence"])
        segs_list.append(seg_list)
    return segs_list



if __name__ == '__main__':
    print("start")
    metadata = pd.read_csv('data/PMEmo_audio_features.csv')
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="0eee6e2465a94368ad5ec7450a2fd62a",
                                               client_secret="35113bd096574a4dae4381bc4090f7ba"))

    """
    musicId,spotifyId,"analysis_sample_rate" "analysis_channels"
    "start": 0.70154,
    "duration": 0.19891,
    "confidence": 0.435,
    "loudness_start": -23.053,
    "loudness_max": -14.25,
    "loudness_max_time": 0.07305,
    "loudness_end": 0,
    "pitches":12
      "timbre": 12
    """
    total_segs = []
    total_secs = []
    for index, row in metadata.iterrows():
        seg_info,sec_info = get_audio_analysis(row['spotifyId'])

        total_segs.extend(seg_info)
        total_secs.extend(sec_info)
        
    print(len(total_segs))
    df = pd.DataFrame(total_secs, columns =['spotifyId','analysis_sample_rate','analysis_sample_channel',"start","duration","confidence","loudness","tempo","tempo_confidence","key","key_confidence",'mode','mode_confidence']) 
    df.to_csv('./data/PMEmo_audio_sections.csv')

    df2 = pd.DataFrame(total_segs, columns=['spotifyId','analysis_sample_rate','analysis_sample_channel',
    'start','duration','confidence',
    'loudness_start','loudness_max','loudness_max_time','loudness_end',
    'pitches_0','pitches_1','pitches_2','pitches_3','pitches_4','pitches_5',
    'pitches_6','pitches_7','pitches_8','pitches_9','pitches_10','pitches_11',
    'timbre_0','timbre_1','timbre_2','timbre_3','timbre_4','timbre_5','timbre_6',
    'timbre_7','timbre_8','timbre_9','timbre_10','timbre_11'])
    df2.to_csv('./data/PMEmo_audio_segments.csv')
   
    
    