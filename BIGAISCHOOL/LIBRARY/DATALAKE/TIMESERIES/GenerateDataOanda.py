from DataGenerator import OandaDataGenerator
from Utils.Data_Utils_Generator import Add_Growing_Column, Add_Diff_CO_Column
from pathlib import Path
from os import fspath
import os.path

dg = OandaDataGenerator()
data_df = dg.GetData("2002-01-01T00:00:00Z","2020-12-09T00:00:00Z","EUR_USD","H12","DF")
data_path = Path(Path(__file__).resolve().parent.parent.parent) / "DATALAKE\DATA\OANDA_EURUSD_H12_CLASS.csv"
data_path_last = fspath(data_path)
data_df = Add_Diff_CO_Column(data_df)
data_df = Add_Growing_Column(data_df)
data_df.to_csv(data_path_last,index=False)
oko = 5