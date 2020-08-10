//+------------------------------------------------------------------+
//|                                                   czas_odnas.mq4 |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
struct _SYSTEMTIME {
  ushort wYear;         // 2014 etc
  ushort wMonth;        // 1 - 12
  ushort wDayOfWeek;    // 0 - 6 with 0 = Sunday
  ushort wDay;          // 1 - 31
  ushort wHour;         // 0 - 23
  ushort wMinute;       // 0 - 59
  ushort wSecond;       // 0 - 59
  ushort wMilliseconds; // 0 - 999
};
#import "kernel32.dll"
void GetSystemTime(_SYSTEMTIME &time);
void GetSystemTimeAsFileTime(ulong &SystemTimeAsFileTime);
#import

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- create timer
   EventSetMillisecondTimer(1);
   
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

   
  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
_SYSTEMTIME st;
  GetSystemTime(st);
string ti = st.wYear+"-";
if (StringLen(st.wMonth) == 1) ti = ti+"0"+st.wMonth+"-";
else ti = ti+st.wMonth+"-";
if (StringLen(st.wDay) == 1) ti = ti+"0"+st.wDay+"T";
else ti = ti+st.wDay+"T";
if (StringLen(st.wHour) == 1) ti = ti+"0"+st.wHour+":";
else ti = ti+st.wHour+":";
if (StringLen(st.wMinute) == 1) ti = ti+"0"+st.wMinute+":";
else ti = ti+st.wMinute+":";
if (StringLen(st.wSecond) == 1) ti = ti+"0"+st.wSecond+":";
else ti = ti+st.wSecond+":";

if (StringLen(st.wMilliseconds) == 1) ti = ti+"00"+st.wMilliseconds;
if (StringLen(st.wMilliseconds) == 2) ti = ti+"0"+st.wMilliseconds;
else ti = ti+st.wMilliseconds;
Print(ti);

for(int i=0;i<200;i++)
{
MqlTick last_tick;
SymbolInfoTick("USDJPY",last_tick);
}
OrderSend(Symbol(),OP_BUY,0.1,Ask,2,Bid-15*Point,Bid+15*Point);

_SYSTEMTIME st2;
 GetSystemTime(st2);
string t2 = st2.wYear+"-";
if (StringLen(st2.wMonth) == 1) t2 = t2+"0"+st2.wMonth+"-";
else t2 = t2+st2.wMonth+"-";
if (StringLen(st2.wDay) == 1) t2 = t2+"0"+st2.wDay+"T";
else t2 = t2+st2.wDay+"T";
if (StringLen(st2.wHour) == 1) t2 = t2+"0"+st2.wHour+":";
else t2 = t2+st2.wHour+":";
if (StringLen(st2.wMinute) == 1) t2 = t2+"0"+st2.wMinute+":";
else t2 = t2+st2.wMinute+":";
if (StringLen(st2.wSecond) == 1) t2 = t2+"0"+st2.wSecond+":";
else t2 = t2+st2.wSecond+":";

if (StringLen(st2.wMilliseconds) == 1) t2 = t2+"00"+st2.wMilliseconds;
if (StringLen(st2.wMilliseconds) == 2) t2 = t2+"0"+st2.wMilliseconds;
else t2 = t2+st2.wMilliseconds;
Print(t2);
   
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
