import os
import numpy as np
import pickle

import pandas as pd
import cv2

t = np.load('/home/k/kzheng3/Final/data/spect.npy')
print(np.max(t/255))
print(t.shape)
