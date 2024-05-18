""" Created on 15th January
Author : Payal Mohapatra
Contact : PayalMohapatra2026@u.northwestern.edu
Project : Person Identification using biosignal signatures

Version 3 :
1. Use raw values of the acc and gyro + stdev
2. Use hour of the day as an input after the conv layers
3. Use the calories as an input after the LSTM layers
4. Smoothen heart rate data to observe longer trends

"""


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
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KernelDensity

## Date-time-parsing 
from datetime import datetime
from dateutil import parser

## Torch related
## Pytorch related
import torch
from torch._C import dtype
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


## Helper functions
def __extract_labels__(user_dir) :
   # Our data is max 47 -- two digits so this is fine 
   ## change in future
   tens_digit = 10
   final_num = 0
   
   for i in user_dir:
      if i.isdigit() :
         final_num += tens_digit*int(i)
         tens_digit -= 9
   return final_num

def __feature_scaling__(min_val, max_val, data) :
   data_norm = (data - min_val) / (max_val - min_val)
   return data_norm

def slice_data(window_length, pd_vitals, overlap_ratio) :
   ## window_length --> if sampling rate is at 1 min per sample and you want 3 hours window --> 180 is window length
   ## pd_vitals --> dataframe
   ## overlap_ratio --> [0,1]
    start = 0
    end = window_length
    slice = pd_vitals[start : end]
    start = start + int(window_length * (1-overlap_ratio))
    end = start + window_length
    while end <= len(pd_vitals) :
        # print(end)
        slice_curr = pd_vitals[start : end]
        slice = np.dstack((slice, slice_curr))
        start = start + int(window_length * (1-overlap_ratio))
        end = start + window_length
    
    if (slice.ndim == 2) : ## If there is only one window's worth of data in a file
        if (isinstance(slice,(np.ndarray)) == False) :
            slice = slice.to_numpy()
        slice_new = np.reshape(slice, (1, window_length, np.shape(slice)[1]))
    else :
        slice_new = slice.transpose(2,0,1)
    return slice_new

## Main script
def __feature_engineering__(training_folder, train_or_valid, window_length, overlap_ratio, sample_period, sample_period_in_sec) :
    ## Obtain the iterator for each user
    for user_iter, user_dir in enumerate(os.listdir(training_folder)) :
        # print('Working on : ', user_dir)
        user_id = __extract_labels__(user_dir)
        print('User ID : ', user_id)
        ## Get the training data from each user for each day
        # Path for each user trainin
        user_dir_path = os.path.join(training_folder, user_dir , train_or_valid)
        for user_day_iter, user_day in enumerate(os.listdir(user_dir_path)) :
          
          print('Working on day : ', user_day)
          user_day_data_path = os.path.join(user_dir_path, user_day, 'data.csv')
          user_day_step_path = os.path.join(user_dir_path, user_day, 'step.csv')
    
          data_df = pd.read_csv(user_day_data_path, parse_dates=['timecol'])
          data_steps_df = pd.read_csv(user_day_step_path, parse_dates=['start_time', 'end_time'])

          if (len(data_steps_df) == 0):
            ## Skip the union
            print('Skipping the steps union.....')
            ## fill dummy values
            data_df['calories']  = 0 
            data_df['distance']  = 0 
            data_df['stepsRunning'] = 0  
            data_df['stepsWalking'] = 0  
            data_df['totalSteps']  = 0 
            data_df['duration']  = 0 
          else :
            data_steps_df['duration'] = (data_steps_df.end_time - data_steps_df.start_time).astype(int)*1e-9 ## This is in seconds
            newindex = data_df.index.union(data_steps_df.index)
            data_df = data_df.join(data_steps_df, how='outer')
    
          ######################## HOUR ONE HOT ENCODING ##############################################
          ## Make an hour column
          hour_tmp_df = pd.to_datetime(data_df.timecol)
          data_df['hour'] = hour_tmp_df.dt.hour
          
          ### one hot encoding
          data_df['hour_0'] = ((data_df.hour >= 4)  & (data_df.hour < 12))
          data_df['hour_1'] = ((data_df.hour >= 12) & (data_df.hour < 16))
          data_df['hour_2'] = ((data_df.hour >= 16) & (data_df.hour < 21))
          data_df['hour_3'] = ((data_df.hour >= 21) | (data_df.hour < 4))
          
          ### convert the bool to integer
          data_df.hour_0 = data_df.hour_0.replace({True: 1, False: 0})
          data_df.hour_1 = data_df.hour_1.replace({True: 1, False: 0})
          data_df.hour_2 = data_df.hour_2.replace({True: 1, False: 0})
          data_df.hour_3 = data_df.hour_3.replace({True: 1, False: 0})
    
          data_df.hour_0 = data_df.hour_0.apply(pd.to_numeric) 
          data_df.hour_1 = data_df.hour_1.apply(pd.to_numeric) 
          data_df.hour_2 = data_df.hour_2.apply(pd.to_numeric) 
          data_df.hour_3 = data_df.hour_3.apply(pd.to_numeric) 
          
          ### drop the hour column
          data_df = data_df.drop(['hour'], axis=1)
          
    
          ######################## CARDIAC FILTERING --> SCALING --> SMOOTHING ##############################################
          ## Filter data 
          data_df = data_df[(data_df['heartRate'] > 30)]
          data_df = data_df[(data_df['rRInterval'] > 0)]
    
          ### range for hr : [30, 255]
          data_df['heartRate'] = __feature_scaling__(30,255, data_df.heartRate)
                
          ### range for rR intervals : [0, 2000]
          data_df['rRInterval'] = __feature_scaling__(0, 2000, data_df.rRInterval)    
    
          ## rolling_window of 5 mins ## FIXME
          rolling_window = int(5*60/sample_period_in_sec)
          data_df['heartRate'] = data_df.loc[:,'heartRate'].rolling(rolling_window).mean() 
    
    
          ######################## IMU FEATURE SCALING --> RMS COLUMN ##############################################
          
          ### range for acc : [-19.6, 19,6]
          data_df['acc_X'] = __feature_scaling__(-19.6, 19.6, data_df.acc_X)
          data_df['acc_Y'] = __feature_scaling__(-19.6, 19.6, data_df.acc_Y)
          data_df['acc_Z'] = __feature_scaling__(-19.6, 19.6, data_df.acc_Z)
                
          ### range for gyro : [-573, 573]
          data_df['gyr_X'] = __feature_scaling__(-573, 573, data_df.gyr_X)
          data_df['gyr_Y'] = __feature_scaling__(-573, 573, data_df.gyr_Y)
          data_df['gyr_Z'] = __feature_scaling__(-573, 573, data_df.gyr_Z)
          
          ### Get the RMS values
          data_df['acc_rms']  = np.sqrt(data_df.acc_X**2 + data_df.acc_Y**2 + data_df.acc_Z**2)
          data_df['gyr_rms'] = np.sqrt(data_df.gyr_X**2 + data_df.gyr_Y**2 + data_df.gyr_Z**2 )
       
          
          
          ########################  FEATURE ENGINEERING FOR STEPS ##############################################
          ## Feature scaling for duration
          max_duration = (window_length * sample_period_in_sec)
          min_duration = 0
          data_df['duration'] = __feature_scaling__(min_duration, max_duration, data_df.duration)
          
    
          ## Feature scaling for calories
          ## I assume 2000 calories can be burned in a day. ~83 calories in an hour -- Take a relaxed upperbound of 100 calories per hour
          max_calories = ((window_length * sample_period_in_sec)/3600) * 100
          min_calories = 0
          data_df['calories'] = __feature_scaling__(min_calories, max_duration, data_df.calories)
    
    
          ## Feature scaling for steps
          # Keep the scaling factor same to maintain linearity
          # Assume ideal fitness of running per hour -- 10 Km per hour --> ~8km per hour may be --> 10,000 steps per hour
          max_steps = ((window_length * sample_period_in_sec)/3600) * 10000
          min_steps = 0
    
          data_df['stepsRunning'] = __feature_scaling__(min_steps, max_steps, data_df.stepsRunning) 
          data_df['stepsWalking'] = __feature_scaling__(min_steps, max_steps, data_df.stepsWalking) 
          data_df['totalSteps']   = __feature_scaling__(min_steps, max_steps, data_df.totalSteps) 
    
          max_distance = 90000 ## distance is in meters --> relaxed upperbound of 9km
          min_distance = 0
          data_df['distance']   = __feature_scaling__(min_distance, max_distance, data_df.distance) 
    
          # Downsample data
          # Currently resampled to 1 minute from 5 seconds
          # Note :: resample removes all object type columns so, if you need to do any operations on timecol, do it prior
          if (len(data_df) > 0) :
            print("Before : ", len(data_df))
            data_df_tmp = data_df
            data_df = data_df_tmp.resample(sample_period, on='timecol').mean()
            data_df_std = data_df_tmp.resample(sample_period, on='timecol').std()
            print("After :", len(data_df_std))
          
            data_df['acc_rms_std'] = data_df_std['acc_rms']
            data_df['gyr_rms_std'] = data_df_std['gyr_rms']
            data_df['heartRate_std'] = data_df_std['heartRate']
            data_df['rRInterval_std'] = data_df_std['rRInterval']   
          
            ## Fill steps values with NaN and then after slicing take the sum and normalize
            data_df['calories']     = data_df['calories'].fillna(0)
            data_df['distance']     = data_df['distance'].fillna(0)
            data_df['stepsRunning'] = data_df['stepsRunning'].fillna(0)
            data_df['stepsWalking'] = data_df['stepsWalking'].fillna(0)
            data_df['totalSteps']   = data_df['totalSteps'].fillna(0)
            data_df['duration']     = data_df['duration'].fillna(0)
                
            data_df['user_id'] = user_id
          
            # ## Drop NaN
            data_df = data_df.dropna()
    
            print('Number of features' , len(data_df.columns))
          
            ## TODO :: update for variable lengths for variable lengths 
            if (len(data_df) >= window_length ) :
              pd_sliced_curr = slice_data(window_length, data_df, overlap_ratio)
              ## Take the cumulative duration for the steps
              ### Get locations --> take sum --> substitute
              pd_sliced_curr[0,:,data_df.columns.get_loc('duration')] = np.sum(pd_sliced_curr[0,:,data_df.columns.get_loc('duration')])
              pd_sliced_curr[0,:,data_df.columns.get_loc('stepsWalking')] = np.sum(pd_sliced_curr[0,:,data_df.columns.get_loc('stepsWalking')])
              pd_sliced_curr[0,:,data_df.columns.get_loc('stepsRunning')] = np.sum(pd_sliced_curr[0,:,data_df.columns.get_loc('stepsRunning')])
              pd_sliced_curr[0,:,data_df.columns.get_loc('distance')] = np.sum(pd_sliced_curr[0,:,data_df.columns.get_loc('distance')])
              pd_sliced_curr[0,:,data_df.columns.get_loc('calories')] = np.sum(pd_sliced_curr[0,:,data_df.columns.get_loc('calories')])
              pd_sliced_curr[0,:,data_df.columns.get_loc('totalSteps')] = np.sum(pd_sliced_curr[0,:,data_df.columns.get_loc('totalSteps')])
    
              ## Normalise duration wrt window_size and sampling rate
              
    
              if ((user_day_iter == 0) & (user_iter == 0)):
                pd_sliced = pd_sliced_curr
              else :
                pd_sliced = np.concatenate((pd_sliced, pd_sliced_curr), axis = 0)
    
            print('Number of segments so far are : ', np.shape(pd_sliced)[0])
    return pd_sliced, data_df ## data_df returned only to get the columns

######################################### PARAMETERS TO CHANGE ################################################

if __name__ == "__main__":
    ####Original is 5 seconds for data.csv
    window_length = 180
    overlap_ratio = 0
    #### Change the below varibales in tandem #####
    sample_period = '1T'
    sample_period_in_sec = 60
    train_or_valid = 'train/'
    ## Change this for your path
    training_folder = '/home/payal/SPGC_challenge_track_1_release/SPGC_challenge_track_1_release/training_data'
    pd_sliced, test_df = __feature_engineering__(training_folder, train_or_valid, window_length, overlap_ratio, sample_period, sample_period_in_sec)

    x_train_acc = pd_sliced[:,:,[test_df.columns.get_loc('acc_X'), test_df.columns.get_loc('acc_Y'), test_df.columns.get_loc('acc_Z')]]
    x_train_gyr = pd_sliced[:,:,[test_df.columns.get_loc('gyr_X'), test_df.columns.get_loc('gyr_Y'), test_df.columns.get_loc('gyr_Z')]]
    x_train_hr = pd_sliced[:,:,[test_df.columns.get_loc('heartRate'), test_df.columns.get_loc('rRInterval'), test_df.columns.get_loc('missing_feature')]]
    x_train_static = pd_sliced[:,:, [test_df.columns.get_loc('sleeping'), test_df.columns.get_loc('totalSteps'), test_df.columns.get_loc('stepsWalking'), test_df.columns.get_loc('stepsRunning'), test_df.columns.get_loc('distance'), test_df.columns.get_loc('calories'), test_df.columns.get_loc('duration'), test_df.columns.get_loc('hour_0'), test_df.columns.get_loc('hour_1'), test_df.columns.get_loc('hour_2'), test_df.columns.get_loc('hour_3')]]
    y_train = pd_sliced[:,:, [test_df.columns.get_loc('user_id')]]


    np.save('saved_var/missing_feature_encoding_27_01/x_Train_acc_30_0_3', x_train_acc)
    np.save('saved_var/missing_feature_encoding_27_01/x_Train_gyr_30_0_3', x_train_gyr)
    np.save('saved_var/missing_feature_encoding_27_01/x_Train_hr_30_0_3', x_train_hr)
    np.save('saved_var/missing_feature_encoding_27_01/x_Train_static_30_0_3', x_train_static)
    np.save('saved_var/missing_feature_encoding_27_01/y_Train_30_0_3', y_train)