from DataGenerator import OandaDataGenerator
dg = OandaDataGenerator()
from DataPreparator import PyTorchDataPreparator
from pathlib import Path
from os import fspath
import os.path
import pandas as pd


def Online_Oanda_To_Pytorch(start, stop, instrument, timeframe, batch_size, time_steps, train_perc, val_perc, test_perc, target):
    data_df = dg.GetData(start,stop,instrument,timeframe,"DF")
    to_remove = ["time","volume"]
    data_df = data_df[data_df.columns.difference(to_remove)]
    pdp = PyTorchDataPreparator(data_df)
    data_prepared = pdp.GetDataPrepared(batch_size,time_steps,train_perc,val_perc,test_perc,target)
    return data_prepared

def Offline_Oanda_To_Pytorch(instrument, timeframe, batch_size, time_steps, train_perc, val_perc, test_perc, target):
    path = "DATALAKE\DATA\OANDA_" + instrument + "_" + timeframe + ".csv"
    data_path = Path(Path(__file__).resolve().parent.parent.parent) / path
    data_path_last = fspath(data_path)
    data_df = pd.read_csv(data_path_last)
    to_remove = ["time","volume"]
    data_df = data_df[data_df.columns.difference(to_remove)]
    pdp = PyTorchDataPreparator(data_df)
    data_prepared = pdp.GetDataPrepared(batch_size,time_steps,train_perc,val_perc,test_perc,target)
    return data_prepared


Offline_Oanda_To_Pytorch("EUR_USD", "H1", 128, 16, 80, 10, 10, "O")