## Import Utilities

import string
import sys
import time
import matplotlib.pyplot as plt
import IPython.display as ipd
import argparse
import math
from tqdm import tqdm 
import random

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st

## Scikit related
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy import integrate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import cross_val_score
from scipy.stats import skew
from scipy.stats import norm, kurtosis
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
# import xgboost as xgb
from sklearn.neighbors import KernelDensity

## Torch related
## Pytorch related
import torch
from torch._C import dtype
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import date
import sys
## Append other paths

sys.path.append('/home/payal/ICASSP-SPGC-Track-1-2023/saved_var/')
# from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
writer = SummaryWriter(f"Training starting on:{date.today()}")
writer = SummaryWriter(comment="ICASSP:model_360_30_percent_overlap_Adam_COSINE")

## Helper functions for NN
def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

from matplotlib.lines import Line2D


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().detach().cpu().numpy())
            max_grads.append(p.grad.abs().max().detach().cpu().numpy())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available is', device)
set_seed(2711)

# Update the path to the data files
x_train_acc    = np.load('saved_var/x_Train_acc_30_0_3.npy', allow_pickle=True).astype(np.float32)
x_train_gyr    = np.load('saved_var/x_Train_gyr_30_0_3.npy', allow_pickle=True).astype(np.float32)
x_train_hr     = np.load('saved_var/x_Train_hr_30_0_3.npy', allow_pickle=True).astype(np.float32)
x_train_static = np.load('saved_var/x_Train_static_30_0_3.npy', allow_pickle=True).astype(np.float32)
y_train        = np.load('saved_var/y_Train_30_0_3.npy', allow_pickle=True).astype(np.float32)
y_train        = y_train[:,0]

print('Shape of x_train_acc', x_train_acc.shape)
print('Shape of x_train_gyro', x_train_gyr.shape)
print('Shape of x_train_hr', x_train_hr.shape)
print('Shape of x_train_static', x_train_static.shape)
print('Shape of y_train', y_train.shape)


folder_path = 'saved_var/valid_data_1h/' # for valid per day data

batch_size = 256
class MTLGenDataset(Dataset) :
    def __init__(self,x_train_acc, x_train_gyr, x_train_hr, x_train_static, y_train) :
        # data loading
        self.x_train_hr = x_train_hr
        self.x_train_acc = x_train_acc
        self.x_train_gyr = x_train_gyr
        self.x_train_static = x_train_static
        self.y = y_train
        self.n_samples = x_train_hr.shape[0]
        
    def __getitem__(self,index) :
        
        return self.x_train_acc[index], self.x_train_gyr[index], self.x_train_hr[index], self.x_train_static[index], self.y[index]

    def __len__(self) :    
        return self.n_samples  

n_samples_train = x_train_acc.shape[0]
sequence_length = x_train_acc.shape[1]
input_size  = x_train_acc.shape[2]

# print('Number of training data', n_samples_train)

train_dataset = MTLGenDataset(x_train_acc, x_train_gyr, x_train_hr, x_train_static, y_train)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=1)

hidden_size = 64 # You can try any size; Experiment with different sizes :: This choice is very poor I think(We have 47 classes)
hidden_size_fc = 64
hidden_size_fc_2 = 64
learning_rate = 1e-3
num_layers = 1

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_fc, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.conv_acc = nn.Sequential(nn.Conv1d(input_size[0], 8, 3, padding='valid'),
                            nn.ReLU(),
                            nn.Conv1d(8, 8, 3, padding='valid'),
                            nn.ReLU(),
                            nn.Conv1d(8, 16, 3, padding='valid'),
                            nn.ReLU(),            
                           )  
        
        self.conv_gyr = nn.Sequential(nn.Conv1d(input_size[1], 8, 3, padding='valid'),
                            nn.ReLU(),
                            nn.Conv1d(8, 8, 3, padding='valid'),
                            nn.ReLU(),
                            nn.Conv1d(8, 16, 3, padding='valid'),
                            nn.ReLU(),            
                           )   
        
        self.conv_hr = nn.Sequential(nn.Conv1d(input_size[2], 8, 3, padding='valid'),
                            nn.ReLU(),
                            nn.Conv1d(8, 8, 3, padding='valid'),
                            nn.ReLU(),
                            nn.Conv1d(8, 16, 3, padding='valid'),
                            nn.ReLU(),            
                           )
                
        self.lstm = nn.LSTM(16*3,hidden_size,num_layers,batch_first=True)

        self.nn = nn.Sequential(nn.Linear(hidden_size+input_size[3], hidden_size_fc),
                            nn.ReLU(),
                            nn.Linear(hidden_size_fc, hidden_size_fc_2),
                            nn.ReLU(),
                            nn.Linear(hidden_size_fc_2, num_classes)       
                           )         

    
    def forward(self, x_acc, x_gyr, x_hr, x_static):
        
        # conv layers
        # print(x_acc.shape)
        x_acc = torch.permute(x_acc,(0,2,1))      
        x_gyr = torch.permute(x_gyr,(0,2,1))         
        x_hr = torch.permute(x_hr,(0,2,1))
        out_acc = self.conv_acc(x_acc)
        out_gyr = self.conv_gyr(x_gyr)
        out_hr = self.conv_hr(x_hr)
        out_acc = torch.permute(out_acc,(0,2,1))      
        out_gyr = torch.permute(out_gyr,(0,2,1))         
        out_hr = torch.permute(out_hr,(0,2,1))
        
        # concatenate all sensors except static
        out = torch.cat((out_acc, out_gyr, out_hr), 2)
        
        # LSTM        
        h0 = Variable(torch.zeros(self.num_layers, out.size(0), hidden_size).to(device))
        c0 = Variable(torch.zeros(self.num_layers, out.size(0), hidden_size).to(device))
        out, _ = self.lstm(out,(h0,c0))        
        # out = out[:, -1, :]
        out = torch.mean(out, dim=1)
        
        # concatenate with static features
        out = torch.cat((out, x_static[:,0,:]), 1)
        
        # nn
        out = self.nn(out)
        
        return out
    
input_size = [x_train_acc.shape[2], x_train_gyr.shape[2], x_train_hr.shape[2], x_train_static.shape[2]]
num_classes = 46

model = LSTM(input_size, hidden_size, hidden_size_fc, num_layers, num_classes).to(device)
print(model)
print('Number of trainable parameters:', sum(p.numel() for p in model.parameters()))

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
############################ Learning rate scheduler ############################
def adjust_learning_rate_cosine_anealing(optimizer, init_lr, epoch, num_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    # print('Learning rate inside adjusting cosine lr = ', cur_lr)

def adjust_learning_rate_warmup_time(optimizer, init_lr, epoch, num_epochs, model_size, warmup):
    """Decay the learning rate based on warmup schedule based on time
    Source :: https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer 
    """
    cur_lr = (model_size ** (-0.5) * min((epoch+1) ** (-0.5), (epoch+1) * warmup ** (-1.5))) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    # print('Learning rate inside adjusting warmup decay lr = ', cur_lr)

def naive_lr_decay(optimizer, init_lr, epoch, num_epochs):
    """
    Make 3 splits in the num_epochs and just use that to decay the lr 
    """
    if (epoch < np.ceil(num_epochs/4)) :
        cur_lr = init_lr
    elif (epoch < np.ceil(num_epochs/2)) :
        cur_lr = 0.5 * init_lr
    else :
        cur_lr = 0.25 * init_lr    

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    # print('Learning rate inside naive decay lr = ', cur_lr)



n_total_steps = len(train_loader)
num_epochs = 2000
#torch.autograd.set_detect_anomaly(True)
for epoch in range(num_epochs):
    n_correct = 0
    train_loss = 0.0
    # naive_lr_decay(optimizer, learning_rate, epoch, num_epochs)
    adjust_learning_rate_cosine_anealing(optimizer, learning_rate, epoch, num_epochs)
    # print('Learning rate = ', optimizer.param_groups)
    for i, (x_acc,x_gyr,x_hr,x_static,labels) in enumerate(train_loader):  
        # resized: [batch size, sequence length, input size]
        x_acc = x_acc.to(device).to(torch.float32)
        x_gyr = x_gyr.to(device).to(torch.float32)
        x_hr = x_hr.to(device).to(torch.float32)
        x_static = x_static.to(device).to(torch.float32)
        labels = labels.to(device).to(torch.long)  
        labels = labels.reshape((len(labels),))

        # Forward pass
        outputs = model(x_acc, x_gyr, x_hr, x_static)
        loss = criterion(outputs, labels)        
        train_loss = (train_loss*i + loss.item())/(i+1)
        
        outputs_acc = nn.Softmax(dim=1)(outputs)
        outputs_acc = torch.argmax(outputs_acc, dim=1)
        outputs_acc = outputs_acc.cpu().detach().numpy()
        labels_acc = labels.cpu().detach().numpy()

        train_acc = balanced_accuracy_score(outputs_acc, labels_acc)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # plot_grad_flow(model.named_parameters())
        # Refer : https://androidkt.com/how-to-apply-gradient-clipping-in-pytorch/ 
        # torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0, norm_type=0.5)
        # torch.nn.utils.clip_grad_value_(model.parameters(),clip_value=1.0)
        

        optimizer.step()
    
    
            
    with torch.no_grad():
        actual_user_id = np.zeros((10000,))
        predicted_user_id = np.zeros((10000,))        

        for count, path in enumerate(os.listdir(folder_path)):
            user_id = path.split('_')[3].split('.')[0]
            # print(count,user_id)
            x_valid = np.load(folder_path+path, allow_pickle=True).astype(np.float32)
            x_valid = torch.from_numpy(x_valid).to(torch.float32).to(device)
            x_val_acc = x_valid[:,:,[0,1,2]]
            x_val_gyr = x_valid[:,:,[3,4,5]]
            x_val_hr = x_valid[:,:,[6,7,19]]
            x_val_static = x_valid[:,:,8:19]
            outputs = model(x_val_acc, x_val_gyr, x_val_hr, x_val_static)
            outputs = nn.Softmax(dim=1)(outputs)
            outputs = torch.argmax(outputs, dim=1)
            outputs = outputs.to('cpu')
            # print(st.mode(outputs)[0].item())
            actual_user_id[count,] = user_id
            predicted_user_id[count,] = st.mode(outputs)[0].item()
            
        actual_user_id = actual_user_id[0:count+1,]
        predicted_user_id = predicted_user_id[0:count+1,]
        per_day_val_acc = balanced_accuracy_score(actual_user_id, predicted_user_id)
         
    writer.add_scalar("Loss per epoch/train", train_loss, epoch)
    writer.add_scalar("Per Segment accuracy per epoch/train", train_acc, epoch)
    writer.add_scalar("Accuracy per epoch/valid", per_day_val_acc, epoch)
    
    if epoch % 1 == 0:
        print (f'Epoch:{epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {per_day_val_acc:.4f}') 
        # check if the saving directory is available
        if not os.path.exists(f'model_chk/'):
            print('Creating the model checkpoint directory.')
            os.makedirs(f'model_chk/') 
        torch.save(model, f'model_chk/epoch_{epoch+1}.pth')