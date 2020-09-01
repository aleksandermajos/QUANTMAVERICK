import zmq
import zmq.asyncio
from Chart import ChartFX
from Strategy import StrategyFXTickArbitrage
import pandas as pd
import json


class Listener:
    def __init__(self):
        self.context = zmq.Context()
        self.mouth = self.context.socket(zmq.PUB)
        self.mouth.bind("tcp://*:2025")
        self.sub = self.context.socket(zmq.SUB)
        self.req = self.context.socket(zmq.REQ)
        self.Charts = list()
        self.TimeLine = pd.DataFrame()
        self.Strategy = {"ArbitrageO1": StrategyFXTickArbitrage("ArbitrageO1")}
        '''
        self.TimeLine = pd()
        okoi = 5
        self.ChartsAPI = list()
        self.ChartsAPI.append(ChartFXCM("EURUSD"))

        self.ChartsFIX = list()
        self.ChartsFIX.append(ChartFIX("EURUSD"))
        '''
        self.newsub = "2027"
        self.newreq = "2028"
        self.oldsub = ""
        self.oldreq = ""

    def listen(self):
        while True:
            '''
            datafix = self.ChartsFIX[0].GetData("EUR/USD")
            if datafix is None or datafix.empty == True:
                print(" ")
            else:
                print(datafix.to_string(header=False,index=False))
            datafxcmapi = self.ChartsAPI[0].GetData("EUR/USD")
            
            print(datafxcmapi.to_string(header=False,index=False))
            '''
            text = ("PORTS AVAILABLE: "+self.newsub+" "+self.newreq)
            b = bytes(text, 'utf-8')
            self.mouth.send(b)
            if self.newsub != self.oldsub:
                self.sub.connect("tcp://localhost:"+self.newsub)
                self.sub.subscribe("")
                self.req.connect("tcp://localhost:"+self.newreq)
            sub_msg = ""
            try:
                sub_msg = self.sub.recv(flags=zmq.NOBLOCK)
            except zmq.Again as e:
                oko = 5
            if len(sub_msg) > 0:
                sub_msg = sub_msg.decode('utf8').replace("}{", ", ")
                my_json = json.loads(sub_msg)
                json.dumps(sub_msg, indent=4, sort_keys=True)

                WindowID = my_json["WindowID"]
                exist = False
                for x in self.Charts:
                    if x.WindowID == WindowID:
                        exist = True
                        data = x.GetData(my_json)
                        self.Strategy['ArbitrageO1'].Decide(self.Charts)
                        if self.TimeLine.empty:
                            self.TimeLine = data
                        else:
                            self.TimeLine = self.TimeLine.append(data, ignore_index=True)
                        if self.TimeLine.shape[0] %100 == 0:
                            self.TimeLine.to_csv('TIMELINE_EURUSD.csv')
                        print(data.to_string(header=False,index=False))
                if exist == True:
                    self.oldsub = self.newsub
                    self.oldreq = self.newreq
                else:
                    self.Charts.append(ChartFX(my_json, self.newsub, self.newreq))
                    self.oldsub = self.newsub
                    self.oldreq = self.newreq
                    self.newsub = str(int(self.newsub)+2)
                    self.newreq = str(int(self.newreq)+2)