import os
import numpy as np
import pickle

import pandas as pd
import cv2

def make_data(save_dir,ids):
    data = []
    for idx in ids:
        print(idx)
        image_path = os.path.join(save_dir, str(idx)+'.png')
        image = cv2.imread(image_path)
        image = image[200:1200,600:3500,:]#cut only spect
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.array(image)
        data.append(image)
    return data
    # print(len(data))
    # pik = open('drive/MyDrive/SMC/audio_analysis_arousal_new.pickle', 'wb')  # opening pickle file with writing access
    # pickle.dump(data, pik)  # saving the pickle file
    # pik.close() 

if __name__ == '__main__':
    print('start')
    trs = pd.read_csv('/home/k/kzheng3/Final/data/pmemo_train.csv')
    vs = pd.read_csv('/home/k/kzheng3/Final/data/pmemo_valid.csv')
    ts = pd.read_csv('/home/k/kzheng3/Final/data/pmemo_test.csv')
    total = pd.concat((trs,vs,ts),axis = 0)['id']
    #total = total.sort_values(by = 'id')['id'].to_numpy()
    print(total)
    save_dir = '/home/k/kzheng3/Final/data/spects'
    data =make_data(save_dir,total)
    save = np.array(data)
    print(save.shape)
    np.save('./spect_cut.npy',save)