from pathlib import Path
from os import fspath
import os.path
import pandas as pd
def Write_Brokers_Opening_Times(Chart):
    data_path = Path(Path(
        __file__).resolve().parent.parent.parent) / "DATA\BROKERS_OPEN_TIMES.csv"
    data_path_last = fspath(data_path)
    column_names = ["Broker","Symbol", "Period", "OpenTime"]
    df = pd.DataFrame(columns=column_names)
    df = df.append({"Broker": Chart.Broker,"Symbol": Chart.Symbol, "Period": Chart.Period, "OpenTime": Chart.OpenTimes[-1]}, ignore_index=True)
    if os.path.isfile(data_path_last):
        df_from_file = pd.read_csv(data_path_last)
        frames = [df_from_file, df]
        result = pd.concat(frames)
        result.to_csv(data_path_last, index=False, header=True)
    else:
        df.to_csv(data_path_last, index=False, header=True)

def Write_Brokers_Closing_Times(Chart):
    data_path = Path(Path(
        __file__).resolve().parent.parent.parent) / "DATA\BROKERS_CLOSE_TIMES.csv"
    data_path_last = fspath(data_path)
    column_names = ["Broker","Symbol", "Period", "CloseTime"]
    df = pd.DataFrame(columns=column_names)
    df = df.append({"Broker": Chart.Broker,"Symbol": Chart.Symbol, "Period": Chart.Period, "CloseTime": Chart.ClosingTimes[-1]}, ignore_index=True)
    if os.path.isfile(data_path_last):
        df_from_file = pd.read_csv(data_path_last)
        frames = [df_from_file, df]
        result = pd.concat(frames)
        result.to_csv(data_path_last, index=False, header=True)
    else:
        df.to_csv(data_path_last, index=False, header=True)