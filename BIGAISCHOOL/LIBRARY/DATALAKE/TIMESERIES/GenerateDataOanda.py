from DataGenerator import OandaDataGenerator
from Utils.Data_Utils_Generator import Add_Growing_Column, Add_Diff_CO_Column
from pathlib import Path
from os import fspath
import os.path

dg = OandaDataGenerator()
pairs = ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CAD", "USD_CHF", "AUD_USD"]
tf = ["M15", "M30", "H1", "H4", "H12"]
for pair in pairs:
    for timeframe in tf:
        data_df = dg.GetData("2002-01-01T00:00:00Z","2020-12-13T00:00:00Z",pair,timeframe,"DF")
        path = "DATALAKE\DATA\OANDA_"+pair+"_"+timeframe+".csv"
        data_path = Path(Path(__file__).resolve().parent.parent.parent) / path
        data_path_last = fspath(data_path)
        data_df = Add_Diff_CO_Column(data_df)
        data_df = Add_Growing_Column(data_df)
        data_df.to_csv(data_path_last,index=False)