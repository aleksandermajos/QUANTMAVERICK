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

def Open_Time_To_New_Chart(Chart):
    data_path = Path(Path(
        __file__).resolve().parent.parent.parent) / "DATA\BROKERS_OPEN_TIMES.csv"
    data_path_last = fspath(data_path)
    if os.path.isfile(data_path_last):
        df_from_file = pd.read_csv(data_path_last)
        lo=df_from_file.loc[df_from_file['Broker'] == Chart.Broker, 'OpenTime']
        if not lo.empty :
            li = lo.tolist()
            return li
        else: return [200]
    else:
        return [200]


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

def Close_Time_To_New_Chart(Chart):
    data_path = Path(Path(
        __file__).resolve().parent.parent.parent) / "DATA\BROKERS_CLOSE_TIMES.csv"
    data_path_last = fspath(data_path)
    if os.path.isfile(data_path_last):
        df_from_file = pd.read_csv(data_path_last)
        lo=df_from_file.loc[df_from_file['Broker'] == Chart.Broker, 'CloseTime']
        if not lo.empty :
            li = lo.tolist()
            return li
        else: return [200]
    else:
        return [200]

def Average_OpenTimes(Chart):
    if Chart.OpenTimes:
        return (sum(Chart.OpenTimes) / len(Chart.OpenTimes)+max(Chart.OpenTimes))/2
    else: return None

def Average_CloseTimes(Chart):
    if Chart.ClosingTimes:
        return (sum(Chart.ClosingTimes) / len(Chart.ClosingTimes) + max(Chart.ClosingTimes)) / 2
    else:
        return None