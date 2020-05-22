import numpy as np
import pandas as pd
import oandapyV20
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import itertools
from torch import Tensor


def csv_to_tensor(data):
    to_remove = ["time", "Diff_CO", "Growing"]
    data = data[data.columns.difference(to_remove)]
    batch_size = 128
    timesteps = 16
    n_timeseries = data.shape[1] - 1
    train_length = 36000
    val_length = 2866
    test_length = 2866
    target = "O"
    X = np.zeros((len(data), timesteps, data.shape[1] - 1))
    y = np.zeros((len(data), timesteps, 1))
    for i, name in enumerate(list(data.columns[:-1])):
        for j in range(timesteps):
            X[:, j, i] = data[name].shift(timesteps - j - 1).fillna(method="bfill")
    for j in range(timesteps):
        y[:, j, 0] = data["O"].shift(timesteps - j - 1).fillna(method="bfill")
    prediction_horizon = 1
    target = data["O"].shift(-prediction_horizon).fillna(method="ffill").values
    X = X[timesteps:]
    y = y[timesteps:]
    target = target[timesteps:]
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

def get_history_oanda(accountID, token, start, stop, instrument, granularity, format):

    path = "./Data/"
    client = oandapyV20.API(token, environment="live")

    params = {
        "from": start,
        "to": stop,
        "granularity": granularity,
        "count": 4500
    }
    with open(path + "/{}.{}".format(instrument, granularity), "w") as OUT:
        lista = []
        for r in InstrumentsCandlesFactory(instrument=instrument, params=params):
            client.request(r)
            lista.append(r.response.get('candles'))

    if format == "DF": return(List_Of_Dict_To_DF(lista))
    if format == "NP": return(Df_To_NumPy(List_Of_Dict_To_DF(lista)))

def List_Of_Dict_To_DF(lista):
    lista = list(itertools.chain.from_iterable(lista))

    for i in range(len(lista)):
        lista[i]["O"] = lista[i]["mid"]["o"]
        lista[i]["H"] = lista[i]["mid"]["h"]
        lista[i]["L"] = lista[i]["mid"]["l"]
        lista[i]["C"] = lista[i]["mid"]["c"]
        del lista[i]["mid"]

    df = pd.DataFrame.from_dict(lista)
    del df["complete"]
    df = df[["time","O","H","L","C","volume"]]
    df = OHLCV_AS_NUMBERS(df)
    return df

def OHLCV_AS_NUMBERS(df):
    df["O"] = df["O"].astype(np.float32)
    df["H"] = df["H"].astype(np.float32)
    df["L"] = df["L"].astype(np.float32)
    df["C"] = df["C"].astype(np.float32)
    df["volume"] = df["volume"].astype(np.float32)
    return df

def Df_To_NumPy(df):
    del df["time"]
    return df.iloc[:,0:].values

def Add_Diff_CO_Column(df):
    df["Diff_CO"] = df.C-df.O
    return df

def Add_Growing_Column(df):
    df["Growing"] = df["Diff_CO"] >= 0
    df.Growing.replace((True, False), (1, -1), inplace=True)
    return df

def Df_To_CSV(df,path,name):
    df.to_csv(path+name, index=True)

def Df_To_NN(df,HowMany):
    return df

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def Array_To_NN(ara,HowManyCandles,Resolution):
    result =[]
    return ara