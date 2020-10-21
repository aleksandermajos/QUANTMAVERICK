from pathlib import Path
from os import fspath
import os.path
import json
import numpy as np
import pandas as pd
from BIGAISCHOOL.LIBRARY.DATALAKE.TIMESERIES.Utils.Data_Utils_Write_Brokers_Time import Write_Brokers_Opening_Times

def Open_Time_To_Existing_Chart(data, Chart):
    sub_msg = data.decode('utf8').replace("}{", ", ")
    my_json = json.loads(sub_msg)
    json.dumps(sub_msg, indent=4, sort_keys=True)
    my_json["time_start"] = np.datetime64(my_json["time_start"])
    my_json["time_stop"] = np.datetime64(my_json["time_stop"])
    diff = my_json["time_stop"] - my_json["time_start"]
    diff = diff / np.timedelta64(1, 'ms')
    Chart.OpenTimes.append(diff)
    Write_Brokers_Opening_Times(Chart)

def Open_Time_To_New_Chart():
    data_path = Path(Path(
        __file__).resolve().parent.parent.parent) / "DATA\BROKERS_OPEN_TIMES.csv"
    data_path_last = fspath(data_path)
    if os.path.isfile(data_path_last):
        df_from_file = pd.read_csv(data_path_last)
        li = df_from_file['OpenTime'].tolist()
        return li
    else:
        return list()


def Close_Time_To_Existing_Chart(data, Chart):
    sub_msg = data.decode('utf8').replace("}{", ", ")
    my_json = json.loads(sub_msg)
    json.dumps(sub_msg, indent=4, sort_keys=True)
    my_json["time_start"] = np.datetime64(my_json["time_start"])
    my_json["time_stop"] = np.datetime64(my_json["time_stop"])
    diff = my_json["time_stop"] - my_json["time_start"]
    diff = diff / np.timedelta64(1, 'ms')
    Chart.CloseTimes.append(diff)
    Write_Brokers_Opening_Times(Chart)

def Close_Time_To_New_Chart():
    data_path = Path(Path(
        __file__).resolve().parent.parent.parent) / "DATA\BROKERS_CLOSE_TIMES.csv"
    data_path_last = fspath(data_path)
    if os.path.isfile(data_path_last):
        df_from_file = pd.read_csv(data_path_last)
        li = df_from_file['CloseTime'].tolist()
        return li
    else:
        return list()