//+------------------------------------------------------------------+
//|                                                     TestyZMQ.mq4 |
//|                                                    QUANTMAVERICK |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "QUANTMAVERICK"
#property link      ""
#property version   "1.00"
#property strict
#include <Zmq/Zmq.mqh>
#include <QUANTMAVERICKMT4/Util.mq4>

Context context;
Socket SUBLISTENERPORTS(context,ZMQ_SUB);
Socket socPUB(context,ZMQ_PUB);
Socket socREP(context,ZMQ_REP);
int SUBPUB_PORT = 0;
int REQREP_PORT = 0;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- create timer
   EventSetTimer(1);
SUBLISTENERPORTS.connect("tcp://localhost:2025");
SUBLISTENERPORTS.subscribe("");
   
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- destroy timer
   EventKillTimer();
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
  ZmqMsg message("");
SUBLISTENERPORTS.recv(message,true);
string Message = message.getData();
Print(Message);

  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
 ZmqMsg message("");
SUBLISTENERPORTS.recv(message,true);
string Message = message.getData();
Print(Message);
   
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   
  }
//+------------------------------------------------------------------+
