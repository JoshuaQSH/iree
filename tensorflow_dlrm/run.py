import numpy as np
import pandas as pd
import sys
import os
import logging
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# Custome LabelEncoding Class 
from src.utils import LabelEncoding, time_count
from src.model import DLRMModel, warmup, train_model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import warnings

warnings.filterwarnings('ignore')
print("TF Version: ", tf.__version__)

# Data Path
data_path = '/home/shenghao/dataset/adult.csv'

# Checking GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

data = pd.read_csv(data_path)
# print(data.shape)
all_columns = data.columns
num_cols = ['fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = list(set(all_columns)-set(num_cols))
target_cols = ['income']

# Set data
num_data = data[num_cols]
cat_data = data[cat_cols]
target = data[target_cols]

target[target_cols] = (target[target_cols]=='>50K').astype(int)
scaler = MinMaxScaler()
num_data_scaled = scaler.fit_transform(num_data)
lbenc = LabelEncoding()
lbenc.fit(cat_data, cat_data.columns)
cat_enc = lbenc.transform(cat_data)
# print(cat_enc.head(2))

# Model Architecture
'''
Numerical Input:
512, 256, 16 with Relu
Embedding: 
16 for each input category
Make 2nd order inner product layer
concatenate all the output from the 2nd order term and numerical values
Classification Layer:
input : 16 * feature_num
512, 256, 1 (sigmoid)
'''

# Datasets X and y
y = target[target_cols].values
X = [num_data.values]

cat_x = []
for col in cat_enc:
    cat_x.append(cat_enc[col].values)
X.append(cat_x)

feature_dic = {}
for col in cat_enc:
    feature_dic[col] = cat_enc[col].nunique()

# Model Define
dlrm = DLRMModel(num_data, cat_enc, feature_dic)
# plot_model(dlrm)

print("Normal TF training...")
dlrm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
warmup(dlrm, X, y)
train_model(dlrm, X, y, 2)

# Using the XLA compiler. To enable the compiler in the middle of the application, 
# we need to reset the Keras session.
print("XLA Training...")
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA
dlrm_xla = DLRMModel(num_data, cat_enc, feature_dic)
dlrm_xla.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
warmup(dlrm_xla, X, y)
train_model(dlrm_xla, X, y, 2) 

# standalone
# dlrm.fit(X, y, epochs=2)
