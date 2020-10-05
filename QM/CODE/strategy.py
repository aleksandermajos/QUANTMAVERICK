from abc import ABC, abstractmethod
import json

class Strategy(ABC):

    def __init__(self):
        super(Strategy, self).__init__()

    @abstractmethod
    def Decide(self, Charts):
        pass


class StrategyFX(Strategy):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def Decide(self, Charts):
        pass


class StrategyFXTickArbitrage(StrategyFX):
    def __init__(self, name):
        self.name = name


    def Decide(self, Charts):
        if len(Charts) > 6:
            for x in Charts:
                for y in Charts:
                    if (x.ap_diff > 0 and y.ap_diff < 0) and ((abs(x.ap_diff)+abs(y.ap_diff)) > 0.00002):
                        message_x = {'key': 'OPEN', 'symbol': x.instrument.name, 'operation':"OP_SELL",'volume': 0.01,
                                     'price': x.actuall["Ask"].iloc[0],'slippage': 0,'TP': 0, 'SL': 0}
                        message_x_json = json.dumps(message_x)

                        message_y = {'key': 'OPEN','symbol': y.instrument.name, 'operation': "OP_BUY", 'volume': 0.01,
                                     'price': y.actuall["Bid"].iloc[0], 'slippage': 0, 'TP': 0, 'SL': 0}
                        message_y_json = json.dumps(message_y)

                        x.req.send(message_x_json.encode('utf-8'))
                        y.req.send(message_y_json.encode('utf-8'))

                        datax = x.req.recv()
                        datay = y.req.recv()

                        oko=6

                    if (x.ap_diff < 0 and y.ap_diff > 0) and ((abs(x.ap_diff) + abs(y.ap_diff)) > 0.00002):
                        message_x = {'key': 'OPEN','symbol': x.instrument.name, 'operation': "OP_BUY", 'volume': 0.01,
                                     'price': x.actuall["Bid"].iloc[0], 'slippage': 0, 'TP': 0, 'SL': 0}
                        message_x_json = json.dumps(message_x)

                        message_y = {'key': 'OPEN','symbol': y.instrument.name, 'operation': "OP_SELL", 'volume': 0.01,
                                     'price': y.actuall["Ask"].iloc[0], 'slippage': 0, 'TP': 0, 'SL': 0}
                        message_y_json = json.dumps(message_y)

                        x.req.send(message_x_json.encode('utf-8'))
                        y.req.send(message_y_json.encode('utf-8'))

                        datax = x.req.recv()
                        datay = y.req.recv()

                        oko=8
        return 0
