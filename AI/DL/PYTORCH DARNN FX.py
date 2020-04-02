import torch
from torch import nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'..')
from Utils.Neural_Nets_Utils import DARNN
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv("./Data/EURUSDM5.csv")
to_remove = ["time","Diff_CO","Growing"]
data = data[data.columns.difference(to_remove)]

batch_size = 128
timesteps = 16
n_timeseries = data.shape[1]-1
train_length = 36000
val_length = 2866
test_length = 2866
target = "O"

X = np.zeros((len(data), timesteps, data.shape[1]-1))
Y = np.zeros((len(data), timesteps, 1))

for i, name in enumerate(list(data.columns[:-1])):
    for j in range(timesteps):
        X[:, j, i] = data[name].shift(timesteps - j - 1).fillna(method="bfill")

for j in range(timesteps):
        Y[:, j, 0] = data["O"].shift(timesteps - j - 1).fillna(method="bfill")

prediction_horizon = 1
target = data["O"].shift(-prediction_horizon).fillna(method="ffill").values

X = X[timesteps:]
Y = Y[timesteps:]
target = target[timesteps:]

X_train = X[:train_length]
X_val = X[train_length:train_length+val_length]
X_test = X[-val_length:]
Y_his_train = Y[:train_length]
Y_his_val = Y[train_length:train_length+val_length]
Y_his_test = Y[-val_length:]
target_train = target[:train_length]
target_val = target[train_length:train_length+val_length]
target_test = target[-val_length:]

X_train_max = X_train.max(axis=0)
X_train_min = X_train.min(axis=0)
Y_his_train_max = Y_his_train.max(axis=0)
Y_his_train_min = Y_his_train.min(axis=0)
target_train_max = target_train.max(axis=0)
target_train_min = target_train.min(axis=0)

X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
X_val = (X_val - X_train_min) / (X_train_max - X_train_min)
X_test = (X_test - X_train_min) / (X_train_max - X_train_min)

Y_his_train = (Y_his_train - Y_his_train_min) / (Y_his_train_max - Y_his_train_min)
Y_his_val = (Y_his_val - Y_his_train_min) / (Y_his_train_max - Y_his_train_min)
Y_his_test = (Y_his_test - Y_his_train_min) / (Y_his_train_max - Y_his_train_min)

target_train = (target_train - target_train_min) / (target_train_max - target_train_min)
target_val = (target_val - target_train_min) / (target_train_max - target_train_min)
target_test = (target_test - target_train_min) / (target_train_max - target_train_min)

X_train_t = torch.Tensor(X_train)
X_val_t = torch.Tensor(X_val)
X_test_t = torch.Tensor(X_test)
Y_his_train_t = torch.Tensor(Y_his_train)
Y_his_val_t = torch.Tensor(Y_his_val)
Y_his_test_t = torch.Tensor(Y_his_test)
target_train_t = torch.Tensor(target_train)
target_val_t = torch.Tensor(target_val)
target_test_t = torch.Tensor(target_test)

model = DARNN(X_train.shape[2], 64, 64, X_train.shape[1]).cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, gamma=0.9)

from torch.utils.data import TensorDataset, DataLoader
data_train_loader = DataLoader(TensorDataset(X_train_t, Y_his_train_t, target_train_t), shuffle=True, batch_size=128)
data_val_loader = DataLoader(TensorDataset(X_val_t, Y_his_val_t, target_val_t), shuffle=False, batch_size=128)
data_test_loader = DataLoader(TensorDataset(X_test_t, Y_his_test_t, target_test_t), shuffle=False, batch_size=128)

epochs = 150
loss = nn.MSELoss()
patience = 15
min_val_loss = 9999
counter = 0
for i in range(epochs):
    mse_train = 0
    for batch_x, batch_y_h, batch_y in data_train_loader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        batch_y_h = batch_y_h.cuda()
        opt.zero_grad()
        y_pred = model(batch_x, batch_y_h)
        y_pred = y_pred.squeeze(1)
        l = loss(y_pred, batch_y)
        l.backward()
        mse_train += l.item() * batch_x.shape[0]
        opt.step()
    epoch_scheduler.step()
    with torch.no_grad():
        mse_val = 0
        preds = []
        true = []
        for batch_x, batch_y_h, batch_y in data_val_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_y_h = batch_y_h.cuda()
            output = model(batch_x, batch_y_h)
            output = output.squeeze(1)
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            mse_val += loss(output, batch_y).item() * batch_x.shape[0]
    preds = np.concatenate(preds)
    true = np.concatenate(true)

    if min_val_loss > mse_val ** 0.5:
        min_val_loss = mse_val ** 0.5
        print("Saving...")
        torch.save(model.state_dict(), "darnn_FX.pt")
        counter = 0
    else:
        counter += 1

    if counter == patience:
        break
    print("Iter: ", i, "train: ", (mse_train / len(X_train_t)) ** 0.5, "val: ", (mse_val / len(X_val_t)) ** 0.5)
    if (i % 10 == 0):
        preds = preds * (target_train_max - target_train_min) + target_train_min
        true = true * (target_train_max - target_train_min) + target_train_min
        mse = mean_squared_error(true, preds)
        mae = mean_absolute_error(true, preds)
        print("mse: ", mse, "mae: ", mae)

model.load_state_dict(torch.load("darnn_FX.pt"))

with torch.no_grad():
    mse_val = 0
    preds = []
    true = []
    for batch_x, batch_y_h, batch_y in data_test_loader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        batch_y_h = batch_y_h.cuda()
        output = model(batch_x, batch_y_h)
        preds.append(output.detach().cpu().numpy())
        true.append(batch_y.detach().cpu().numpy())
        mse_val += loss(output, batch_y).item()*batch_x.shape[0]
preds = np.concatenate(preds)
true = np.concatenate(true)

preds = preds*(target_train_max - target_train_min) + target_train_min
true = true*(target_train_max - target_train_min) + target_train_min

mse = mean_squared_error(true, preds)
mae = mean_absolute_error(true, preds)