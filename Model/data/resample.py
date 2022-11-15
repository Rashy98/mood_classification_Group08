import librosa    
import os
from scipy.io.wavfile import write
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import resampy


if __name__ == '__main__':
    files = []
    for f in os.listdir('/home/k/kzheng3/MER/medieval/vocal_only'):
        if f.endswith(".wav"):
            x, sr_orig = librosa.load(os.path.join('/home/k/kzheng3/MER/medieval/vocal_only',f), sr=None)
            y_low = resampy.resample(x, sr_orig, 16000,filter="kaiser_best")*32767
            print(f)
            write(os.path.join('/home/k/kzheng3/MER/medieval/vocal_resampled',f), 16000, y_low.astype(np.int16))
       