import os
import cv2
import numpy as np 
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GPU = False
if torch.cuda.is_available():
    # torch.cuda.device_count -> for counting number of gpus. most laptops will have just 1
    device = torch.device("cuda:0")# currently only supporting 1 gpu
    GPU = True
    print('running on GPU')
else:
    device = torch.device("cpu")
    GPU = False
    print("running on CPU")

def fwd_pass(net,X,Y,optimizer,loss_function,train=False):
    if train:
        net.zero_grad()
    output = net(X)
    if(output.shape != Y.shape):
        print("output shape does not match target shape!")
        print("input shape:",X.shape)
        print("output shape:",output.shape)
        print("target shape:",Y.shape)
        exit()
    loss = loss_function(output,Y)
    output = None
    del output
    if train:
        loss.backward()
        optimizer.step()
    return loss

def fit(net,X,Y,train_log,optimizer,loss_function,validation_set,BATCH_SIZE,EPOCHS):
    val_size = int(validation_set*len(X))
    data_size = len(X)
    train_size = data_size - val_size

    for epochs in range(EPOCHS):
        #insample data
        train_average_loss = 0
        val_average_loss = 0
        train_counter = 0
        val_counter = 0
        optimizer = optim.Adam(net.parameters(),lr = 0.001)
        loss_function = nn.MSELoss()
        for i in tqdm(range(0,train_size, BATCH_SIZE ) ):
            batch_X = (X[i:i+BATCH_SIZE]).to(device)
            batch_Y = (Y[i:i+BATCH_SIZE]).to(device)
            train_loss = fwd_pass(net,batch_X,batch_Y,optimizer,loss_function,train=True)
            batch_X = None
            del batch_X
            batch_Y = None
            del batch_Y
            if i%100==0:
                train_average_loss += float(train_loss.cpu())
                train_counter += 1
            train_loss = None
            del train_loss
        #outsample data
        del optimizer,loss_function
        torch.cuda.empty_cache()

        optimizer = optim.Adam(net.parameters(),lr = 0.001)
        loss_function = nn.MSELoss()
        for i in tqdm(range(train_size,data_size,BATCH_SIZE)):
            batch_X = (X[i:i+BATCH_SIZE]).to(device)
            batch_Y = (Y[i:i+BATCH_SIZE]).to(device)
            val_loss = fwd_pass(net,batch_X,batch_Y,optimizer,loss_function,train=False)
            batch_X = None
            del batch_X
            batch_Y = None
            del batch_Y
            if i%10==0:
                val_average_loss += float(val_loss.cpu())
                val_counter += 1
            val_loss = None
            del val_loss
            # print('val loss: ',float(val_loss))
        torch.cuda.empty_cache()
        if(train_counter==0):
            train_counter = 1
        if(val_counter ==0):
            val_counter = 1
        train_log.append([train_average_loss/train_counter,val_average_loss/val_counter]) # just store the last values for now

        optimizer = None
        loss_function = None

        del optimizer, loss_function
        torch.cuda.empty_cache()
    return train_log       




