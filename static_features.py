import sys
import copy
import os
import torch
import numpy as np
import pandas as pd
from einops import rearrange
from Utilities_A_B_C import *
import wandb


import torch.optim as optim

# Define different optimizers and hyperparameters
optimizers = {
    'Adam_0.1': lambda params: optim.Adam(params, lr=0.1),
    'Adam_0.01': lambda params: optim.Adam(params, lr=0.01),
    'Adam_0.001': lambda params: optim.Adam(params, lr=0.001),
    #'SGD_0.1_M0.9': lambda params: optim.SGD(params, lr=0.1, momentum=0.9),
    #'SGD_0.01_M0.9': lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
    'SGD_0.001_M0.9': lambda params: optim.SGD(params, lr=0.001, momentum=0.9),
    #'RMSprop_0.1': lambda params: optim.RMSprop(params, lr=0.1),
    'RMSprop_0.01': lambda params: optim.RMSprop(params, lr=0.01),
    'RMSprop_0.001': lambda params: optim.RMSprop(params, lr=0.001),
    #'AdamW_0.1_WD0.01': lambda params: optim.AdamW(params, lr=0.1, weight_decay=0.01),
    #'AdamW_0.01_WD0.01': lambda params: optim.AdamW(params, lr=0.01, weight_decay=0.01),
    'AdamW_0.001_WD0.01': lambda params: optim.AdamW(params, lr=0.001, weight_decay=0.01),
    'Adagrad_0.1': lambda params: optim.Adagrad(params, lr=0.1),
    'Adagrad_0.01': lambda params: optim.Adagrad(params, lr=0.01),
    'Adagrad_0.001': lambda params: optim.Adagrad(params, lr=0.001)
}

use_wandb = True
reset_project=False
# Initialize W&B project
if use_wandb:
    wandb.login()
    wandb_project = "static_features"
    # Delete all runs in the project
    api = wandb.Api()
    if reset_project:
        for run in api.runs(wandb_project):
            run.delete()

#Parameters

NPERIODS    = [10, 20, 30]
EPOC       = int(1000)    # Epochs

#MN = ['Linear', 'CNN', 'LSTM', 'GRU', 'Transformer']
MN = ['Linear', 'CNN', 'LSTM', 'GRU']
NM   = len(MN)    # Number of Models
df = pd.read_csv('all_paths_continue_A_B_C.csv')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
TRAIN_LOSS = np.zeros((NM, 3), dtype=float)   # Training Loss for Route A, B, C
TEST_LOSS  = np.zeros((NM, 3), dtype=float)   # Testing  Loss for Route A, B, C
FEA_TYPE   = 0                   
FEA_APPEND = 0                   
Weights_Directory = 'static_feature_weigths/'


class Linear_Net(torch.nn.Module):
    def __init__(self, n_time, n_output):
        super(Linear_Net, self).__init__()
        self.FL1o1 = torch.nn.Linear(n_time*3, 32)    # First 3 dynamic features
        self.FL1o2 = torch.nn.Linear(n_time*3, 32)           # Static features (5,6,7)
        self.FL1o3 = torch.nn.Linear(n_time*3, 32)    # Dynamic features (3,4,8)
        self.FL1o4 = torch.nn.Linear(n_time*1, 32)    # Last feature (9)
        
        self.FL2o1, self.FL2o2, self.FL2o3, self.FL2o4 = [torch.nn.Linear(32,16) for _ in range(4)]
        self.FL3o1, self.FL4o1 = torch.nn.Linear(16, 3), torch.nn.Linear(3, 8)
        self.FL3o2, self.FL4o2 = torch.nn.Linear(16, 3), torch.nn.Linear(3, 8)
        self.FL3o3, self.FL4o3 = torch.nn.Linear(16, 3), torch.nn.Linear(3, 8)
        self.FL3o4, self.FL4o4 = torch.nn.Linear(16, 1), torch.nn.Linear(1, 8)
        self.FL5o1, self.FL5o2, self.FL5o3, self.FL5o4 = [torch.nn.Linear(8,n_output) for _ in range(4)]
        
        self.SL1 = torch.nn.Linear(n_output*4, 16)
        self.SL2 = torch.nn.Linear(16, 4)
        self.SL3 = torch.nn.Linear(4, n_output)
        self.LRelu = torch.nn.LeakyReLU()

    def forward(self, x):
        x1 = rearrange(x[:,:,:3], 'b t f -> b (t f)')      # First 3 dynamic features
        static = rearrange(x[:,:,5:8], 'b t f -> b (t f)')                                 # Static features (5,6,7)
        x2 = rearrange(x[:,:,[3,4,8]], 'b t f -> b (t f)') # Dynamic features (3,4,8)
        x3 = rearrange(x[:,:,9:], 'b t f -> b (t f)')      # Last feature (9)

        x1 = self.FL5o1(self.LRelu(self.FL4o1(self.LRelu(self.FL3o1(self.LRelu(self.FL2o1(torch.sigmoid(self.FL1o1(x1)))))))))
        static = self.FL5o2(self.LRelu(self.FL4o2(self.LRelu(self.FL3o2(self.LRelu(self.FL2o2(torch.sigmoid(self.FL1o2(static)))))))))
        x2 = self.FL5o3(self.LRelu(self.FL4o3(self.LRelu(self.FL3o3(self.LRelu(self.FL2o3(torch.sigmoid(self.FL1o3(x2)))))))))
        x3 = self.FL5o4(self.LRelu(self.FL4o4(self.LRelu(self.FL3o4(self.LRelu(self.FL2o4(torch.sigmoid(self.FL1o4(x3)))))))))

        x = torch.cat((x1, static, x2, x3), 1)
        x = self.LRelu(self.SL1(x))
        x = self.LRelu(self.SL2(x))
        x = self.SL3(x)
        return x, x1, static, x2, x3
    
class CNN_Net(torch.nn.Module):
    def __init__(self, n_time, n_output):
        super(CNN_Net, self).__init__()
        self.CV1 = torch.nn.Conv1d(3, 3, 3, stride=1, padding='same')
        self.CV2 = torch.nn.Conv1d(3, 3, 3, stride=1, padding='same')
        self.CV3 = torch.nn.Conv1d(3, 3, 3, stride=1, padding='same')
        self.CV4 = torch.nn.Conv1d(1, 1, 3, stride=1, padding='same')
        
        self.FL1o1 = torch.nn.Linear(n_time*3, 32)
        self.FL1o2 = torch.nn.Linear(n_time*3, 32)
        self.FL1o3 = torch.nn.Linear(n_time*3, 32)
        self.FL1o4 = torch.nn.Linear(n_time*1, 32)
        
        self.FL2o1, self.FL2o2, self.FL2o3, self.FL2o4 = [torch.nn.Linear(32,16) for _ in range(4)]
        self.FL3o1, self.FL4o1 = torch.nn.Linear(16, 3), torch.nn.Linear(3, 8)
        self.FL3o2, self.FL4o2 = torch.nn.Linear(16, 3), torch.nn.Linear(3, 8)
        self.FL3o3, self.FL4o3 = torch.nn.Linear(16, 3), torch.nn.Linear(3, 8)
        self.FL3o4, self.FL4o4 = torch.nn.Linear(16, 1), torch.nn.Linear(1, 8)
        self.FL5o1, self.FL5o2, self.FL5o3, self.FL5o4 = [torch.nn.Linear(8,n_output) for _ in range(4)]

        self.SL1 = torch.nn.Linear(n_output*4, 16)
        self.SL2 = torch.nn.Linear(16, 4)
        self.SL3 = torch.nn.Linear(4, n_output)
        self.LRelu = torch.nn.LeakyReLU()

    def forward(self, x):
        # Split and process features
        x1 = rearrange(x[:,:,:3], 'b t f -> b f t')
        static = rearrange(x[:,:,5:8], 'b t f -> b f t')
        x2 = rearrange(x[:,:,[3,4,8]], 'b t f -> b f t')
        x3 = rearrange(x[:,:,9:], 'b t f -> b f t')

        # Apply convolutions to dynamic features
        x1 = torch.sigmoid(self.CV1(x1))
        static = torch.sigmoid(self.CV2(static))
        x2 = torch.sigmoid(self.CV3(x2))
        x3 = torch.sigmoid(self.CV4(x3))

        x1 = rearrange(x1, 'b f t -> b (t f)')
        static = rearrange(static, 'b f t -> b (t f)')
        x2 = rearrange(x2, 'b f t -> b (t f)')
        x3 = rearrange(x3, 'b f t -> b (t f)')

        # Process through fully connected layers
        x1 = self.FL5o1(self.LRelu(self.FL4o1(self.LRelu(self.FL3o1(self.LRelu(self.FL2o1(torch.sigmoid(self.FL1o1(x1)))))))))
        static = self.FL5o2(self.LRelu(self.FL4o2(self.LRelu(self.FL3o2(self.LRelu(self.FL2o2(torch.sigmoid(self.FL1o2(static)))))))))
        x2 = self.FL5o3(self.LRelu(self.FL4o3(self.LRelu(self.FL3o3(self.LRelu(self.FL2o3(torch.sigmoid(self.FL1o3(x2)))))))))
        x3 = self.FL5o4(self.LRelu(self.FL4o4(self.LRelu(self.FL3o4(self.LRelu(self.FL2o4(torch.sigmoid(self.FL1o4(x3)))))))))

        x = torch.cat((x1, static, x2, x3), 1)
        x = self.LRelu(self.SL1(x))
        x = self.LRelu(self.SL2(x))
        x = self.SL3(x)
        return x, x1, static, x2, x3
    
class LSTM_Net(torch.nn.Module):
    def __init__(self, n_time, n_output):
        super(LSTM_Net, self).__init__()
        self.LS1 = torch.nn.LSTM(3, 3, 2, bidirectional=False, batch_first=True)
        self.LS2 = torch.nn.LSTM(3, 3, 2, bidirectional=False, batch_first=True)
        self.LS3 = torch.nn.LSTM(3, 3, 2, bidirectional=False, batch_first=True)
        self.LS4 = torch.nn.LSTM(1, 1, 2, bidirectional=False, batch_first=True)
        
        self.FL1o1 = torch.nn.Linear(n_time*3, 32)    # First 3 dynamic features
        self.FL1o2 = torch.nn.Linear(n_time*3, 32)           # Static features (5,6,7)
        self.FL1o3 = torch.nn.Linear(n_time*3, 32)    # Dynamic features (3,4,8)
        self.FL1o4 = torch.nn.Linear(n_time*1, 32)    # Last feature (9)
        
        self.FL2o1, self.FL2o2, self.FL2o3, self.FL2o4 = [torch.nn.Linear(32,16) for _ in range(4)]
        self.FL3o1, self.FL4o1 = torch.nn.Linear(16, 3), torch.nn.Linear(3, 8)
        self.FL3o2, self.FL4o2 = torch.nn.Linear(16, 3), torch.nn.Linear(3, 8)
        self.FL3o3, self.FL4o3 = torch.nn.Linear(16, 3), torch.nn.Linear(3, 8)
        self.FL3o4, self.FL4o4 = torch.nn.Linear(16, 1), torch.nn.Linear(1, 8)
        self.FL5o1, self.FL5o2, self.FL5o3, self.FL5o4 = [torch.nn.Linear(8,n_output) for _ in range(4)]

        self.SL1 = torch.nn.Linear(n_output*4, 16)
        self.SL2 = torch.nn.Linear(16, 4)
        self.SL3 = torch.nn.Linear(4, n_output)
        self.LRelu = torch.nn.LeakyReLU()

    def forward(self, x):
        x1 = torch.sigmoid(self.LS1(x[:,:,:3])[0])           # First 3 dynamic features
        static = torch.sigmoid(self.LS2(x[:,:,5:8])[0])                                  # Static features (5,6,7)
        x2 = torch.sigmoid(self.LS3(x[:,:,[3,4,8]])[0])     # Dynamic features (3,4,8)
        x3 = torch.sigmoid(self.LS4(x[:,:,9:])[0])          # Last feature (9)

        x1 = rearrange(x1, 'b t f -> b (t f)')
        static = rearrange(static, 'b t f -> b (t f)')
        x2 = rearrange(x2, 'b t f -> b (t f)')
        x3 = rearrange(x3, 'b t f -> b (t f)')

        x1 = self.FL5o1(self.LRelu(self.FL4o1(self.LRelu(self.FL3o1(self.LRelu(self.FL2o1(torch.sigmoid(self.FL1o1(x1)))))))))
        static = self.FL5o2(self.LRelu(self.FL4o2(self.LRelu(self.FL3o2(self.LRelu(self.FL2o2(torch.sigmoid(self.FL1o2(static)))))))))
        x2 = self.FL5o3(self.LRelu(self.FL4o3(self.LRelu(self.FL3o3(self.LRelu(self.FL2o3(torch.sigmoid(self.FL1o3(x2)))))))))
        x3 = self.FL5o4(self.LRelu(self.FL4o4(self.LRelu(self.FL3o4(self.LRelu(self.FL2o4(torch.sigmoid(self.FL1o4(x3)))))))))

        x = torch.cat((x1, static, x2, x3), 1)
        x = self.LRelu(self.SL1(x))
        x = self.LRelu(self.SL2(x))
        x = self.SL3(x)
        return x, x1, static, x2, x3


class GRU_Net(torch.nn.Module):
    def __init__(self, n_time, n_output):
        super(GRU_Net, self).__init__()
        self.GRU1 = torch.nn.GRU(3, 3, 2, bidirectional=False, batch_first=True)
        self.GRU2 = torch.nn.GRU(3, 3, 2, bidirectional=False, batch_first=True)
        self.GRU3 = torch.nn.GRU(3, 3, 2, bidirectional=False, batch_first=True)
        self.GRU4 = torch.nn.GRU(1, 1, 2, bidirectional=False, batch_first=True)
        
        self.FL1o1 = torch.nn.Linear(n_time*3, 32)    # First 3 dynamic features
        self.FL1o2 = torch.nn.Linear(n_time*3, 32)           # Static features (5,6,7)
        self.FL1o3 = torch.nn.Linear(n_time*3, 32)    # Dynamic features (3,4,8)
        self.FL1o4 = torch.nn.Linear(n_time*1, 32)    # Last feature (9)
        
        self.FL2o1, self.FL2o2, self.FL2o3, self.FL2o4 = [torch.nn.Linear(32,16) for _ in range(4)]
        self.FL3o1, self.FL4o1 = torch.nn.Linear(16, 3), torch.nn.Linear(3, 8)
        self.FL3o2, self.FL4o2 = torch.nn.Linear(16, 3), torch.nn.Linear(3, 8)
        self.FL3o3, self.FL4o3 = torch.nn.Linear(16, 3), torch.nn.Linear(3, 8)
        self.FL3o4, self.FL4o4 = torch.nn.Linear(16, 1), torch.nn.Linear(1, 8)
        self.FL5o1, self.FL5o2, self.FL5o3, self.FL5o4 = [torch.nn.Linear(8,n_output) for _ in range(4)]

        self.SL1 = torch.nn.Linear(n_output*4, 16)
        self.SL2 = torch.nn.Linear(16, 4)
        self.SL3 = torch.nn.Linear(4, n_output)
        self.LRelu = torch.nn.LeakyReLU()

    def forward(self, x):
        x1 = torch.sigmoid(self.GRU1(x[:,:,:3])[0])           # First 3 dynamic features
        static = torch.sigmoid(self.GRU2(x[:,:,5:8])[0])                                    # Static features (5,6,7)
        x2 = torch.sigmoid(self.GRU3(x[:,:,[3,4,8]])[0])      # Dynamic features (3,4,8)
        x3 = torch.sigmoid(self.GRU4(x[:,:,9:])[0])           # Last feature (9)

        x1 = rearrange(x1, 'b t f -> b (t f)')
        static = rearrange(static, 'b t f -> b (t f)')
        x2 = rearrange(x2, 'b t f -> b (t f)')
        x3 = rearrange(x3, 'b t f -> b (t f)')

        x1 = self.FL5o1(self.LRelu(self.FL4o1(self.LRelu(self.FL3o1(self.LRelu(self.FL2o1(torch.sigmoid(self.FL1o1(x1)))))))))
        static = self.FL5o2(self.LRelu(self.FL4o2(self.LRelu(self.FL3o2(self.LRelu(self.FL2o2(torch.sigmoid(self.FL1o2(static)))))))))
        x2 = self.FL5o3(self.LRelu(self.FL4o3(self.LRelu(self.FL3o3(self.LRelu(self.FL2o3(torch.sigmoid(self.FL1o3(x2)))))))))
        x3 = self.FL5o4(self.LRelu(self.FL4o4(self.LRelu(self.FL3o4(self.LRelu(self.FL2o4(torch.sigmoid(self.FL1o4(x3)))))))))

        x = torch.cat((x1, static, x2, x3), 1)
        x = self.LRelu(self.SL1(x))
        x = self.LRelu(self.SL2(x))
        x = self.SL3(x)
        return x, x1, static, x2, x3


class Transformer(torch.nn.Module):
    def __init__(self, n_time, n_output):
        super(Transformer, self).__init__()
        self.TF1 = torch.nn.Transformer(d_model=3,  nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16, batch_first=True)
        self.TF2 = torch.nn.Transformer(d_model=6, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16, batch_first=True)
        self.TF3 = torch.nn.Transformer(d_model=1,  nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16, batch_first=True)
        self.FL1o1 = torch.nn.Linear(n_time*3,  32)
        self.FL1o2 = torch.nn.Linear(n_time*6, 32)
        self.FL1o3 = torch.nn.Linear(n_time*1,  32)
        self.FL2o1, self.FL2o2, self.FL2o3 = torch.nn.Linear(32,16), torch.nn.Linear(32,16), torch.nn.Linear(32,16)
        self.FL3o1, self.FL4o1 = torch.nn.Linear(16,  3), torch.nn.Linear(3,  8)
        self.FL3o2, self.FL4o2 = torch.nn.Linear(16, 6), torch.nn.Linear(6, 8)
        self.FL3o3, self.FL4o3 = torch.nn.Linear(16,  1), torch.nn.Linear(1,  8)
        self.FL5o1, self.FL5o2, self.FL5o3 = torch.nn.Linear(8,n_output), torch.nn.Linear(8,n_output), torch.nn.Linear(8,n_output)

        self.SL1 = torch.nn.Linear(n_output*3, 16)
        self.SL2 = torch.nn.Linear(16, 4)
        self.SL3 = torch.nn.Linear(4,  n_output)
        self.LRelu = torch.nn.LeakyReLU()
    def forward(self, x):
        x1 = torch.sigmoid(self.TF1(x[:,:,:3],  x[:,:,:3]  ))
        x2 = torch.sigmoid(self.TF2(x[:,:,3:9],x[:,:,3:9]))
        x3 = torch.sigmoid(self.TF3(x[:,:,9:], x[:,:,9:] ))
        x1 = rearrange(x1, 'b f t -> b (t f)')
        x2 = rearrange(x2, 'b f t -> b (t f)')
        x3 = rearrange(x3, 'b f t -> b (t f)')
        x1 = self.FL5o1( self.LRelu(self.FL4o1( self.LRelu(self.FL3o1( self.LRelu(self.FL2o1( torch.sigmoid(self.FL1o1(x1)))))))))
        x2 = self.FL5o2( self.LRelu(self.FL4o2( self.LRelu(self.FL3o2( self.LRelu(self.FL2o2( torch.sigmoid(self.FL1o2(x2)))))))))
        x3 = self.FL5o3( self.LRelu(self.FL4o3( self.LRelu(self.FL3o3( self.LRelu(self.FL2o3( torch.sigmoid(self.FL1o3(x3)))))))))

        x = torch.cat((x1, x2, x3), 1)
        x = self.LRelu(self.SL1(x))
        x = self.LRelu(self.SL2(x))
        x = self.SL3(x)
        return x, x1, x2, x3

    
# train for different NPERIOD
for NPERIOD in NPERIODS:
    for im, model_name in enumerate(MN):
        for opt_name, opt_fn in optimizers.items():
            route = 'A'
            trainx, trainy, trainf, MAV_trainy, testx, testy, NF, BI, BN, MEAN, STD = Load_Data(df[df['route']==route], 0, 30, 5, NPERIOD, FEA_APPEND, FEA_TYPE, device,0,0,0)
            print('----- Route A -----')
            print(f'========== {model_name} | NPERIOD={NPERIOD} | Optimizer={opt_name} ==========')


            if   im==0:   net = Linear_Net( n_time=NPERIOD, n_output=1)
            elif im==1:   net = CNN_Net(    n_time=NPERIOD, n_output=1)
            elif im==2:   net = LSTM_Net(   n_time=NPERIOD, n_output=1)
            elif im==3:   net = GRU_Net(    n_time=NPERIOD, n_output=1)
            elif im==4:   net = Transformer(n_time=NPERIOD, n_output=1)
            net.to(device)

            if use_wandb:
                # Initialize W&B for this model
                wandb.init(
                    project=wandb_project,
                    name=f"{model_name}-NPERIOD-{NPERIOD}-{opt_name}-Route-{route}",
                    config={"NPERIOD": NPERIOD, "Model": model_name, "Optimizer": opt_name, "Route": route},
                    reinit=True
                )

                # Log model architecture
                wandb.watch(net, log="all")

            BL, PATH  = np.inf, Weights_Directory+model_name+'-'+route+'_'+str(opt_name)+'---'+str(FEA_APPEND)+'_'+str(FEA_TYPE)+'_'+str(NPERIOD)+'_'+str(EPOC)
            loss_fn   = torch.nn.L1Loss()
            optimizer = opt_fn(net.parameters())
            for epoc in range(EPOC):
                LP0, LP1, LP2, LP3, LPstatic = 0, 0, 0, 0, 0
                for b in range(BN):
                    t, f = BI[b], BI[b+1]
                    pred, pred1, static, pred2, pred3 = net(trainx[t:f])
                    optimizer.zero_grad()
                    loss = loss_fn(torch.reshape(pred ,(-1,)), trainy[t:f])
                    LP0 += float(loss)
                    loss.backward(retain_graph=True)
                    loss = loss_fn(torch.reshape(pred1,(-1,)), trainy[t:f])/4.
                    LP1 += float(loss)
                    loss.backward(retain_graph=True)
                    loss = loss_fn(torch.reshape(static,(-1,)), trainy[t:f])/4.
                    LPstatic += float(loss)
                    loss.backward(retain_graph=True)
                    loss = loss_fn(torch.reshape(pred2,(-1,)), trainy[t:f])/4.
                    LP2 += float(loss)
                    loss.backward(retain_graph=True)
                    loss = loss_fn(torch.reshape(pred3,(-1,)), trainy[t:f])/4.
                    LP3 += float(loss)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                pred, pred1, static, pred2, pred3 = net(trainx)
                loss = float(loss_fn(torch.reshape(pred,(-1,)), trainy))

                # # Logging to W&B
                if use_wandb:
                    wandb.log({
                        "epoch": epoc + 1,
                        "train_loss": loss,
                        "LP0": LP0,
                        "LP1": LP1,
                        "LP2": LP2,
                        "LP3": LP3,
                        "LPstatic": LPstatic
                 })

                if epoc%10==9: 
                    SUM = LP0+LP1+LP2+LP3+LPstatic
                    print(epoc+1, '/', EPOC, '  ', round(float(loss), 5), round(LP0,1), round(LP1,1), round(LP2,1), round(LP3,1), round(LPstatic,1), ' - ', round(LP0/SUM,3), round(LP1/SUM,3), round(LP2/SUM,3), round(LP3/SUM,3), round(LPstatic/SUM,3))
                if loss<BL:
                    BL = loss
                    torch.save(net.state_dict(), PATH+'.pt') 
                    print('Save Best at', epoc, 'with loss of', loss)
                # go to next model if the loss is NaN
                if np.isnan(loss):
                    print(f'Loss is NaN at epoch {epoc}. Skipping to next model.')
                    #delete the weights file
                    if os.path.exists(PATH+'.pt'):
                        os.remove(PATH+'.pt')
                    if use_wandb:
                        #delete the wandb run
                        wandb.finish()
                    break
            #try if the weights was saved, else just pass
            try:
                net.load_state_dict(torch.load(PATH+'.pt'))
                Net  = copy.deepcopy(net)
                pred, pred1, static, pred2, pred3 = net(trainx)
                TRAIN_LOSS[im][0] = float(loss_fn(torch.reshape(pred,(-1,)), trainy))
                print('Training Loss', round(TRAIN_LOSS[im][0],3) )
                pred, pred1, static, pred2, pred3 = net(testx)
                np.save(PATH+'.npy', pred.cpu().detach().numpy().reshape((-1,)))
                TEST_LOSS[im][0] = float(loss_fn(torch.reshape(pred,(-1,)), testy))
                print('Testing Loss',  round(TEST_LOSS[ im][0],3) )

                # Log the testing loss to W&B
                if use_wandb:
                    wandb.log({
                        "Testing Loss": TEST_LOSS[im][0],
                        "Training Loss": TRAIN_LOSS[im][0]
                    })
            except FileNotFoundError:
                print(f'Error: Model file {PATH}.pt not found. The model may not have been saved.')
            except RuntimeError as e:
                print(f'Error loading model state: {e}')
            except Exception as e:
                print(f'Unexpected error: {e}')
            
            #Finish Wandb logging for this model
            if use_wandb:
                wandb.finish()
            
                
            for ir, route in enumerate(['B', 'C']):
                print('----- Route', route, ' -----')

                if use_wandb:
                    # Initialize W&B for this model
                    wandb.init(
                        project=wandb_project,
                        name=f"{model_name}-NPERIOD-{NPERIOD}-{opt_name}-Route-{route}",
                        config={"NPERIOD": NPERIOD, "Model": model_name, "Optimizer": opt_name, "Route": route},
                        reinit=True
                    )

                    # Log model architecture
                    wandb.watch(net, log="all")

                if ir==0:   F,T=30,58
                elif ir==1: F,T=58,72
                trainx, trainy, trainf, MAV_trainy, testx, testy, NF, BI, BN, mean, std = Load_Data(df[df['route']==route], F, T, 5, NPERIOD, FEA_APPEND, FEA_TYPE, device,0,0,0)
                net = copy.deepcopy(Net)
                net.to(device)
                BL, PATH  = np.inf, Weights_Directory+model_name+'-'+route+'_'+str(opt_name)+'---'+str(FEA_APPEND)+'_'+str(FEA_TYPE)+'_'+str(NPERIOD)+'_'+str(EPOC)
                loss_fn   = torch.nn.L1Loss()
                optimizer = opt_fn(net.parameters())
                for epoc in range(EPOC):
                    LP0, LP1, LP2, LP3, LPstatic = 0, 0, 0, 0, 0
                    for b in range(BN):
                        t, f = BI[b], BI[b+1]
                        pred, pred1, static, pred2, pred3 = net(trainx[t:f])
                        optimizer.zero_grad()
                        loss = loss_fn(torch.reshape(pred ,(-1,)), trainy[t:f])
                        LP0 += float(loss)
                        loss.backward(retain_graph=True)
                        loss = loss_fn(torch.reshape(pred1,(-1,)), trainy[t:f])/4.
                        LP1 += float(loss)
                        loss.backward(retain_graph=True)
                        loss = loss_fn(torch.reshape(static,(-1,)), trainy[t:f])/4.
                        LPstatic += float(loss)
                        loss.backward(retain_graph=True)
                        loss = loss_fn(torch.reshape(pred2,(-1,)), trainy[t:f])/4.
                        LP2 += float(loss)
                        loss.backward(retain_graph=True)
                        loss = loss_fn(torch.reshape(pred3,(-1,)), trainy[t:f])/4.
                        LP3 += float(loss)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    pred, pred1, static, pred2, pred3 = net(trainx)
                    loss = float(loss_fn(torch.reshape(pred,(-1,)), trainy))
                    
                    
                    
                    
                    # # Logging to W&B
                    if use_wandb:
                        wandb.log({
                            "epoch": epoc + 1,
                            "train_loss": loss,
                            "LP0": LP0,
                            "LP1": LP1,
                            "LP2": LP2,
                            "LP3": LP3,
                            "LPstatic": LPstatic
                    })
                    
                    
                    if epoc%10==9: 
                        SUM = LP0+LP1+LP2+LP3+LPstatic
                        print(epoc+1, '/', EPOC, '  ', round(float(loss), 5), round(LP0,1), round(LP1,1), round(LP2,1), round(LP3,1), round(LPstatic,1), ' - ', round(LP0/SUM,3), round(LP1/SUM,3), round(LP2/SUM,3), round(LP3/SUM,3), round(LPstatic/SUM,3))
                    if loss<BL:
                        BL = loss
                        torch.save(net.state_dict(), PATH+'.pt') 
                        print('Save Best at', epoc, 'with loss of', loss)
                    # go to next model if the loss is NaN
                    if np.isnan(loss):
                        print(f'Loss is NaN at epoch {epoc}. Skipping to next model.')
                        #delete the weights file
                        if os.path.exists(PATH+'.pt'):
                            os.remove(PATH+'.pt')
                        if use_wandb:
                            #delete the wandb run
                            wandb.finish()
                        break
                
                try:
                    net.load_state_dict(torch.load(PATH+'.pt'))
                    Net = copy.deepcopy(net)
                    pred, pred1, static, pred2, pred3 = net(trainx)
                    TRAIN_LOSS[im][ir+1] = float(loss_fn(torch.reshape(pred,(-1,)), trainy))
                    print('Training Loss', round(TRAIN_LOSS[im][ir+1],3) )
                    pred, pred1, static, pred2, pred3 = net(testx)
                    np.save(PATH+'.npy', pred.cpu().detach().numpy().reshape((-1,)))
                    TEST_LOSS[im][ir+1] = float(loss_fn(torch.reshape(pred,(-1,)), testy))
                    print('Testing Loss',  round(TEST_LOSS[ im][ir+1],3) )

                    # Log the testing loss to W&B
                    if use_wandb:
                        wandb.log({
                            "Testing Loss": TEST_LOSS[im][ir+1],
                            "Training Loss": TRAIN_LOSS[im][ir+1]
                        })
                except FileNotFoundError:
                    print(f'Error: Model file {PATH}.pt not found. The model may not have been saved.')
                except RuntimeError as e:
                    print(f'Error loading model state: {e}')
                except Exception as e:
                    print(f'Unexpected error: {e}')
            
            
                #Finish Wandb logging for this model
                if use_wandb:
                    wandb.finish()
            
            torch.cuda.empty_cache()

print('A (Source)', 'B(Transfer)', 'C (Transfer)')
for im, model_name in enumerate(MN):
    print(model_name+':')
    print('Train:', end=' ')
    for i in range(3):
        print(round(TRAIN_LOSS[im][i],5), end=', ')
    print()
    print('Test :', end=' ')
    for i in range(3):
        print(round(TEST_LOSS[im][i],5), end=', ')
    print()


np.save(Weights_Directory+'1-2StagePP_'+str(NPERIOD)+'_'+str(EPOC)+'.npy', [TRAIN_LOSS, TEST_LOSS])