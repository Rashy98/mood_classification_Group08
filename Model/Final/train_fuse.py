import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error,make_scorer,r2_score, mean_absolute_error


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import time
import pickle
import argparse
from tqdm import tqdm

from dataset import fusecnnlmDataset
from arch.fuse_cnnmlp import fuse_cs_direct
import random

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def compare_res(pred,gt):
    prob = torch.exp(pred)
    pred = torch.argmax(prob,axis=1)
    return torch.sum(torch.eq(pred,gt)).item()

def train(model, args,trainset_loader,validset_loader):
    optimizer = optim.Adam(model.parameters(),args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    best_model_id = -1
    min_valid_loss = float("inf")
    epoch_num = int(args.epoch_num)
    valid_every_k_epoch = 1

    device = args.device
        
    model.to(device)
    train_losses = []
    valid_losses = []
    best_acc = 0
    for epoch in range(epoch_num):
        model.train()
        epoch_train_loss = 0
        for step, (data,labels) in enumerate(tqdm(trainset_loader)):
            x1 = data[0].to(device)
            x2 = data[1].to(device)

            label = labels.type(torch.LongTensor).to(device)

            # Zero the gradients
            optimizer.zero_grad()

            logits = model(x1,x2)

            loss = criterion(logits,label)
            loss.backward()
            optimizer.step()

            epoch_train_loss+=loss.detach().item()

        avg_train_loss = epoch_train_loss / len(trainset_loader)
        print('\n', 'Epoch '+str(epoch)+'train loss : ' , str(avg_train_loss))
        train_losses.append(avg_train_loss)

        if(epoch+1)% valid_every_k_epoch == 0:
            epoch_valid_loss = 0
            model.eval()
            cc_count = 0
            with torch.no_grad():
                for vstep, (data,labels) in enumerate(tqdm(validset_loader)):
                    x1 = data[0].to(device)
                    x2 = data[1].to(device)
                    label = labels.type(torch.LongTensor).to(device)
                    # Zero the gradients
                    optimizer.zero_grad()

                    logits = model(x1,x2)

                    log_logits = nn.functional.log_softmax(logits)
                    cc_count = cc_count + compare_res(log_logits,label)

                    vloss = criterion(logits,label)
                    epoch_valid_loss +=vloss.item()

            avg_val_loss = epoch_valid_loss/len(validset_loader)
            print('\n', 'Epoch ',  epoch , ' Val loss : ' , avg_val_loss)
            valid_losses.append(avg_val_loss)
            acc = cc_count/len(validset_loader.dataset)
            print(acc)


            if avg_val_loss<min_valid_loss:#if acc>best_acc:
                best_acc = acc
                min_valid_loss = avg_val_loss
                best_model_id = epoch
                best_stat = model.state_dict()
                torch.save(model.state_dict(), os.path.join(args.save_model_dir,args.best_model_path))
                print('\n', 'Best Epoch ', str(epoch))  
            
    fig = plt.figure()
    x1 = np.arange(epoch_num)
    x2 = np.arange(epoch_num/valid_every_k_epoch)*valid_every_k_epoch+valid_every_k_epoch
    plt.plot(x1,train_losses)
    plt.plot(x2,valid_losses)
    fig.savefig('temp_cnnlm.png')
    
    print('\n', 'Best Epoch ', str(best_model_id),'\n','Min Loss: ', str(min_valid_loss))  
    return best_model_id, best_stat

def predict(args,model, testset_loader):
    device = args.device
    model.to(args.device)
    model.eval()
    preds = []
    gt = []
    with torch.no_grad():
        for vstep, (data,labels) in enumerate(tqdm(testset_loader)):
            x1 = data[0].to(device)
            x2 = data[1].to(device)
            label = labels.type(torch.LongTensor).to(args.device)

            vlogits = model(x1,x2)

            log_logits = nn.functional.log_softmax(vlogits)
            prob = torch.exp(log_logits)
            pred = torch.argmax(prob,axis=1)
            preds.extend(pred.detach().cpu().numpy().tolist())
            gt.extend(label.detach().cpu().numpy().tolist())
    return gt,preds

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model_dir', default='./results', help='path to save the trained models')
    parser.add_argument('--batch_size', default=1, help='batch size for train and validation')
    parser.add_argument('--epoch_num', default=500, help='number of train epochs')
    parser.add_argument('--device', default='cuda', help='cuda device')
    parser.add_argument('--learning_rate', default=1e-3, help='learning rate')
    parser.add_argument('--best_model_path', default='fusecnnlm_marousal.pth')
    parser.add_argument('--load_model', default=False)
    parser.add_argument('--emo_dim', default='arousal')
    parser.add_argument('--is_test', default=False)

    args = parser.parse_args()

    set_seeds(21)

    meta_train = pd.read_csv('./data/pmemo_train.csv')
    meta_test = pd.read_csv('./data/pmemo_test.csv')
    meta_valid = pd.read_csv('./data/pmemo_valid.csv')
    if args.emo_dim == 'valence':
        target_cols = ['id','valence_bin']
    else :
        target_cols = ['id','arousal_bin']

    spect_x = np.load('./data/spect_cut.npy')/255
    spect_x = np.transpose(spect_x, (0,3, 1, 2))
    sptf_x = np.load('./data/spotify/combined_features_normalized.npy')
    print(args.emo_dim)


    train_set = fusecnnlmDataset(spect_x,sptf_x,meta_train[target_cols],0)
    valid_set = fusecnnlmDataset(spect_x,sptf_x,meta_valid[target_cols],len(meta_train))
    test_set = fusecnnlmDataset(spect_x,sptf_x,meta_test[target_cols],len(meta_train)+len(meta_valid))

    train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True,drop_last=True,num_workers=0 )
    valid_dataloader = DataLoader(valid_set,batch_size = 16,shuffle =False,num_workers=0 )
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False,num_workers=0 )

    model = fuse_cs_direct()
    # if args.emo_dim=='arousal':
    #     model.spect_model.load_state_dict(torch.load('./pretrained/cnn_arousal.pth'),strict=False)
    # else:
    #     model.spect_model.load_state_dict(torch.load('./pretrained/cnn_valence.pth'),strict=False)
    model.train()

    if args.is_test:
        best_stat = torch.load(os.path.join(args.save_model_dir,args.best_model_path))
    else:
        best_id,best_stat = train(model,args,train_dataloader,valid_dataloader)
    
    model.load_state_dict(best_stat)
    gt, preds = predict(args,model,test_dataloader)
    df = pd.DataFrame({'gt':gt,'pred':preds})
    df.to_csv('fuse_spotify_spect_'+args.emo_dim+'_test.csv',index = False)
    tacc = np.sum(np.array(gt)==np.array(preds))/len(meta_test)
    gt, preds = predict(args,model,valid_dataloader)
    df = pd.DataFrame({'gt':gt,'pred':preds})
    df.to_csv('fuse_spotify_spect_'+args.emo_dim+'_valid.csv',index = False)
    acc = np.sum(np.array(gt)==np.array(preds))/len(meta_valid)
    gt, preds = predict(args,model,train_dataloader)
    tracc = np.sum(np.array(gt)==np.array(preds))/len(meta_train)
    print(meta_valid[target_cols[1]].to_numpy())
    print('train: ',tracc,'valid: ',acc,' test: ',tacc)

