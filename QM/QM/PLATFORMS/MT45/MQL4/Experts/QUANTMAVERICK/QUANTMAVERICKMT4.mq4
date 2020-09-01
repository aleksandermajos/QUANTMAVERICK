//+------------------------------------------------------------------+
//|                                             QUANTMAVERICKMT4.mq4 |
//|                                Copyright 2019, QuantMaverick Ltd.|
//|                                                                  |
//+------------------------------------------------------------------+

#property strict
#include <Zmq/Zmq.mqh>
#include <QUANTMAVERICKMT4/Util.mq4>

Context context;
Socket socPUB(context,ZMQ_PUB);
Socket socREP(context,ZMQ_REP);

int SUBPUB_PORT = 2027;
int REQREP_PORT = 2028;
bool NEWPORTS = false;

bool NewPorts(string uMessage)
{
if(StringFind(uMessage,"NEWPORTS",0) != -1) return true;
else return false;
}
string ReqRepReply(string Message) 
{
    string Result = "";
    if(StringFind(Message,"CHARTNAME",0) != -1) Result = Result+ChartName(Symbol(), Period(), ChartID());
    if(StringFind(Message,"ACCOUNTINFO",0) != -1) Result = Result+AccountInformation();
    if(StringFind(Message,"INSTRUMENTINFO",0) != -1) Result = Result+InstrumentInfo(Symbol());
    if(StringFind(Message,"HISTORY",0) != -1) Result = Result+History(Symbol());
    if(StringFind(Message,"NEWPORTS",0) != -1) Result = "NEW PORTS SET";
    return Result;
    
}
   


//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- create timer
   EventSetTimer(1);
SUBPUB_PORT = 2027;
REQREP_PORT = 2028;
socPUB.bind("tcp://*:"+SUBPUB_PORT);
socREP.bind("tcp://*:"+REQREP_PORT);



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
  string Message = (TerminalCompany()+"|"+SUBPUB_PORT+"|"+REQREP_PORT);
  ZmqMsg message(Message);

socPUB.send(message);
Print(Message);

 ZmqMsg request;
socREP.recv(request,true);
 
Message = request.getData();
if(Message == NULL) Message = "";
if(Message != "")
{
Print("Received NEW message : "+Message);
ZmqMsg reply(ReqRepReply(Message));
socREP.send(reply);
}
}
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
  string Message = (TerminalCompany()+"|"+SUBPUB_PORT+"|"+REQREP_PORT);
  ZmqMsg message(Message);

socPUB.send(message);
Print(Message);

 ZmqMsg request;
socREP.recv(request,true);
 
Message = request.getData();
if(Message == NULL) Message = "";
if(Message != "")
{
Print("Received NEW message : "+Message);
ZmqMsg reply(ReqRepReply(Message));
socREP.send(reply);
}
   
}
//+------------------------------------------------------------------+
//| Tester function                                                  |
//+------------------------------------------------------------------+
double OnTester()
  {
//---
   double ret=0.0;
//---

//---
   return(ret);
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
