from Utils.Credentials_Utils import exampleAuth
from Utils.Data_Utils_Generator import get_history_oanda


class OandaDataGenerator():
    def __init__(self):
        self.accountID, self.token = exampleAuth('C:\\oanda\\')

    def GetData(self,start, stop, instrument, period, format):
        self.start = start
        self.stop = stop
        self.instrument = instrument
        self.period = period
        self.format = format
        return get_history_oanda(self.accountID, self.token, self.start, self.stop , self.instrument, self.period ,
                                 self.format)




