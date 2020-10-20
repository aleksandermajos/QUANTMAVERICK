from pathlib import Path
from os import fspath
import json
import numpy as np
from BIGAISCHOOL.LIBRARY.DATALAKE.TIMESERIES.Utils.Data_Utils_Write_Brokers_Time import Write_Brokers_Times

def Open_Time_ToChart(data, Chart):
    sub_msg = data.decode('utf8').replace("}{", ", ")
    my_json = json.loads(sub_msg)
    json.dumps(sub_msg, indent=4, sort_keys=True)
    my_json["time_start"] = np.datetime64(my_json["time_start"])
    my_json["time_stop"] = np.datetime64(my_json["time_stop"])
    diff = my_json["time_stop"] - my_json["time_start"]
    diff = diff / np.timedelta64(1, 'ms')
    Chart.OpenTimes.append(diff)
    Write_Brokers_Times(Chart)

def Close_Time_ToChart(data, Chart):
    sub_msg = data.decode('utf8').replace("}{", ", ")
    my_json = json.loads(sub_msg)
    json.dumps(sub_msg, indent=4, sort_keys=True)
    my_json["time_start"] = np.datetime64(my_json["time_start"])
    my_json["time_stop"] = np.datetime64(my_json["time_stop"])
    diff = my_json["time_stop"] - my_json["time_start"]
    diff = diff / np.timedelta64(1, 'ms')
    Chart.CloseTimes.append(diff)
    Write_Brokers_Times(Chart)