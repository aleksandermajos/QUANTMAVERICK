import torch
from torch import nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib as plt
from QM.DATA.DataGenerator import OandaDataGenerator
dg = OandaDataGenerator()
from QM.DATA.DataPreparator import PyTorchDataPreparator
import sys, os
parent_dir = os.getcwd()
path = os.path.dirname(parent_dir)
sys.path.append(path)
sys.path.append('C:\\Users\\hapir\\PycharmProjects\\quantmaverick\\QM\\BIGAISCHOOL\\DEEP LEARNING\\ARCHITECTURES')
from DARNN import DARNN
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = dg.GetData("2018-01-01T00:00:00Z","2020-02-14T00:00:00Z","EUR_USD","H12","DF")
to_remove = ["time","volume"]
data = data[data.columns.difference(to_remove)]
pdp = PyTorchDataPreparator(data)
X_train,target_train_max,target_train_min, X_train_t, X_val_t, X_test_t, y_his_train_t, y_his_val_t, y_his_test_t, \
target_train_t, target_val_t, target_test_t, data_train_loader, data_val_loader, data_test_loader = pdp.GetDataPrepared(128,16,80,10,10,"O")
model = DARNN(X_train.shape[2], 64, 64, X_train.shape[1]).cuda()








