from torch import Tensor
import numpy as np

def df_to_tensor(data, batch_size, time_steps, train_perc, val_perc, test_perc , target):
    n_timeseries = data.shape[1] - 1
    train_length = int(data.shape[0]*train_perc/100)
    val_length = int(data.shape[0]*val_perc/100)
    test_length = int(data.shape[0]*test_perc/100)

    X = np.zeros((len(data), time_steps, data.shape[1] - 1))
    y = np.zeros((len(data), time_steps, 1))
    for i, name in enumerate(list(data.columns[:-1])):
        for j in range(time_steps):
            X[:, j, i] = data[name].shift(time_steps - j - 1).fillna(method="bfill")
    for j in range(time_steps):
        y[:, j, 0] = data[target].shift(time_steps - j - 1).fillna(method="bfill")
    prediction_horizon = 1
    target = data[target].shift(-prediction_horizon).fillna(method="ffill").values
    X = X[time_steps:]
    y = y[time_steps:]
    target = target[time_steps:]
    X_train = X[:train_length]
    X_val = X[train_length:train_length + val_length]
    X_test = X[-val_length:]
    y_his_train = y[:train_length]
    y_his_val = y[train_length:train_length + val_length]
    y_his_test = y[-val_length:]
    target_train = target[:train_length]
    target_val = target[train_length:train_length + val_length]
    target_test = target[-val_length:]
    X_train_max = X_train.max(axis=0)
    X_train_min = X_train.min(axis=0)
    y_his_train_max = y_his_train.max(axis=0)
    y_his_train_min = y_his_train.min(axis=0)
    target_train_max = target_train.max(axis=0)
    target_train_min = target_train.min(axis=0)
    X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
    X_val = (X_val - X_train_min) / (X_train_max - X_train_min)
    X_test = (X_test - X_train_min) / (X_train_max - X_train_min)

    y_his_train = (y_his_train - y_his_train_min) / (y_his_train_max - y_his_train_min)
    y_his_val = (y_his_val - y_his_train_min) / (y_his_train_max - y_his_train_min)
    y_his_test = (y_his_test - y_his_train_min) / (y_his_train_max - y_his_train_min)

    target_train = (target_train - target_train_min) / (target_train_max - target_train_min)
    target_val = (target_val - target_train_min) / (target_train_max - target_train_min)
    target_test = (target_test - target_train_min) / (target_train_max - target_train_min)

    X_train_t = Tensor(X_train)
    X_val_t = Tensor(X_val)
    X_test_t = Tensor(X_test)
    y_his_train_t = Tensor(y_his_train)
    y_his_val_t = Tensor(y_his_val)
    y_his_test_t = Tensor(y_his_test)
    target_train_t = Tensor(target_train)
    target_val_t = Tensor(target_val)
    target_test_t = Tensor(target_test)

    return X_train,target_train_max,target_train_min, X_train_t, X_val_t, X_test_t, y_his_train_t, y_his_val_t, y_his_test_t, target_train_t, target_val_t, target_test_t