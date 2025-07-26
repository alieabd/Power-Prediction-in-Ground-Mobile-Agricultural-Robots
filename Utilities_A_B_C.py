import copy
import torch
import numpy as np
from itertools import chain
from einops import rearrange


def MS(Input):
    Mean, Std = [] ,[]
    Input = rearrange(Input, 'r t f -> f t r')
    for F in range(np.shape(Input)[0]):
        Mean.append(torch.mean(Input[F]))
        Std.append( torch.std( Input[F]))
    return Mean, Std

def Normal(Input, Mean, Std):
    X = rearrange(Input, 'r t f -> f t r')
    for F in range(np.shape(X)[0]):
        X[F] -= Mean[F]
        X[F] /= Std[ F]
    return rearrange(X, 'f t r -> r t f')


# Data, # Featrue Periods, Feature Append, Feature Type
def Decompose(Data, NP, FA, FT):
    x, yv, yc, yp = [], [], [], []
    for i in range(np.shape(Data)[0]):
        row = Data[i].tolist()
        #row.pop(-3)                          # Pop Route
        if   FT==0:   start, to =  3, 13     # All      Features
        elif FT==1:   start, to = 5, 10     # Motor    Features
        elif FT==2:   start, to =  2, 4     # Position Features
        v, c = row[4], row[5]
        yv.append(v)                         # Append Label - Voltage
        yc.append(c)                         # Append Label - Current
        yp.append(row[-1])                       # Append Label - Power
        if   FA==2:   x.append([              v, c, v*c])    # Append Features
        elif FA==1:   x.append(row[start:to]+[v, c, v*c])
        elif FA==0:   x.append(row[start:to]) 
    X, YV, YC, YP, YF = [], [], [], [], []
    for i in range(len(yp)-NP):
        #print(x)
        X.append(  x[i:i+NP])
        YF.append( x[i+NP])
        YV.append(yv[i+NP])
        YC.append(yc[i+NP])
        YP.append(yp[i+NP])
    return X, YV, YC, YP, YF


# Dataframe, Test ID from F to T, Number of Testing Data, # Featrue Periods, Feature Append, Feature Type
# Device, Flag of Mean and Std, Mean, Std
def Load_Data(DF, F, T, NT, NP, FA, FT, device, MSFlag,mean,std):     
    Row = []
    for i in range(F, T):
        tmp = DF[DF['test']==i].values
        if np.shape(tmp)[0]>100: Row.append(copy.deepcopy(tmp))
    train_x, train_yp, train_yf = [], [], []     # train_yv, train_yc, 
    test_x,  test_yp            = [], []         # test_yv,  test_yc,  
    for i in range(len(Row)-NT):
        tmp_x, tmp_yv, tmp_yc, tmp_yp, tmp_yf = Decompose(Row[i], NP, FA, FT)
        train_x.append( tmp_x )     # train_yv.append(tmp_yv)   train_yc.append(tmp_yc)
        train_yp.append(tmp_yp)
        train_yf.append(tmp_yf)
    for i in range(len(Row)-NT,len(Row)):
        tmp_x, tmp_yv, tmp_yc, tmp_yp, tmp_yf = Decompose(Row[i], NP, FA, FT)
        test_x.append( tmp_x )      # test_yv.append(tmp_yv)   test_yc.append(tmp_yc)
        test_yp.append(tmp_yp)
        
    trainx = torch.FloatTensor( list(chain(*train_x )) ).to(device)
    trainy = torch.FloatTensor( list(chain(*train_yp)) ).to(device)
    trainf = torch.FloatTensor( list(chain(*train_yf)) ).to(device)
    testx  = torch.FloatTensor( list(chain(*test_x )) ).to(device)
    testy  = torch.FloatTensor( list(chain(*test_yp)) ).to(device)
    MAV_trainy = float(torch.abs(torch.mean(trainy)))
    if MSFlag==1:   MEAN, STD = mean,std
    else:           MEAN, STD = MS(trainx)
    print('Mean:', MEAN, '  STD:', STD)
    trainx = Normal(trainx, MEAN, STD)
    if NT!=0:   testx  = Normal(testx, MEAN, STD)
        
    NF = np.shape(train_x[0])[2]
    BI, BN = [0], len(train_yp)   # Batch ID
    for i in range(BN):
        BI.append(BI[-1]+len(train_yp[i]))
    print('Number of Features:', NF, '  Train:', len(train_x), '  Test:', len(test_x))
    #print(train_x)
    return trainx, trainy, trainf, MAV_trainy, testx, testy, NF, BI, BN, MEAN, STD