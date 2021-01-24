from .OHLC_Manipulate import Add_Growing_Column, Add_Diff_CO_Column, List_Of_Dict_To_DF, Df_To_NumPy, Df_Remove_Columns
from .DF_to_PyTorch_Tensor import PyTorchDataPreparator
from pathlib import Path
from os import fspath
import pandas as pd
import oandapyV20
from oandapyV20.contrib.factories import InstrumentsCandlesFactory


def exampleAuth(path):
    accountID, token = None, None
    with open(path + "account.txt") as I:
        accountID = I.read().strip()
    with open(path + "token.txt") as I:
        token = I.read().strip()
    return accountID, token


class OandaData():
    def __init__(self):
        self.accountID, self.token = exampleAuth('C:\\oanda\\')

    def GetData(self,start, stop, instrument, period, format):
        self.start = start
        self.stop = stop
        self.instrument = instrument
        self.period = period
        self.format = format
        return self.get_history_oanda(self.accountID, self.token, self.start, self.stop , self.instrument, self.period ,
                                 self.format)

    def GenerateData(self):
        pairs = ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CAD", "USD_CHF", "AUD_USD"]
        tf = ["M15", "M30", "H1", "H4", "H12"]
        for pair in pairs:
            for timeframe in tf:
                data_df = self.GetData("2002-01-01T00:00:00Z", "2020-12-13T00:00:00Z", pair, timeframe, "DF")
                path = "DATALAKE\DATA\OANDA_" + pair + "_" + timeframe + ".csv"
                data_path = Path(Path(__file__).resolve().parent.parent.parent) / path
                data_path_last = fspath(data_path)
                data_df = Add_Diff_CO_Column(data_df)
                data_df = Add_Growing_Column(data_df)
                data_df.to_csv(data_path_last, index=False)

    def Online_Oanda_To_Pytorch(self,start, stop, instrument, timeframe, batch_size, time_steps, train_perc, val_perc,
                                test_perc, target):
        data_df = self.GetData(start, stop, instrument, timeframe, "DF")
        to_remove = ["time", "volume"]
        data_df = Df_Remove_Columns(data_df, to_remove)
        pdp = PyTorchDataPreparator(data_df)
        data_prepared = pdp.GetDataPrepared(batch_size, time_steps, train_perc, val_perc, test_perc, target)
        return data_prepared

    def Offline_Oanda_To_Pytorch(instrument, timeframe, batch_size, time_steps, train_perc, val_perc, test_perc,
                                 target):
        path = "DATALAKE\DATA\OANDA_" + instrument + "_" + timeframe + ".csv"
        data_path = Path(Path(__file__).resolve().parent.parent.parent) / path
        data_path_last = fspath(data_path)
        data_df = pd.read_csv(data_path_last)
        to_remove = ["time", "volume"]
        data_df = Df_Remove_Columns(data_df, to_remove)
        pdp = PyTorchDataPreparator(data_df)
        data_prepared = pdp.GetDataPrepared(batch_size, time_steps, train_perc, val_perc, test_perc, target)
        return data_prepared

    def get_history_oanda(accountID, token, start, stop, instrument, granularity, format):

        client = oandapyV20.API(token, environment="practice")

        params = {
            "from": start,
            "to": stop,
            "granularity": granularity,
            "count": 4500
        }

        lista = []
        for r in InstrumentsCandlesFactory(instrument=instrument, params=params):
            client.request(r)
            lista.append(r.response.get('candles'))

        if format == "DF": return List_Of_Dict_To_DF(lista)
        if format == "NP": return Df_To_NumPy(List_Of_Dict_To_DF(lista))






