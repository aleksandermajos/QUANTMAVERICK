import sys, os
parent_dir = os.getcwd()
path = os.path.dirname(parent_dir)
sys.path.append(path)
sys.path.append('/QM/BIGAISCHOOL/MACHINE LEARNING/DEEP LEARNING/ARCHITECTURES')
sys.path.append('/QM/BIGAISCHOOL/MACHINE LEARNING/DEEP LEARNING/TRAINING')
sys.path.append('/QM/BIGAISCHOOL/MACHINE LEARNING/PYTHON/VISUALIZATIONS')
sys.path.append('/QM/DATA')
from DARNN import DARNN
from Train import Train_DARNN
from DataGenerator import OandaDataGenerator
dg = OandaDataGenerator()
from DataPreparator import PyTorchDataPreparator
from Visualize import candle_OHLC
import plotly.graph_objects as go


data_df = dg.GetData("2010-01-01T00:00:00Z","2020-02-14T00:00:00Z","EUR_USD","H12","DF")
to_remove = ["time","volume"]
data_df = data_df[data_df.columns.difference(to_remove)]
fig = candle_OHLC(data_df)
fig.show()
pdp = PyTorchDataPreparator(data_df)
data_prepared = pdp.GetDataPrepared(128,16,80,10,10,"O")

model = DARNN(data_prepared['X_train'].shape[2], 64, 64, data_prepared['X_train'].shape[1]).cuda()
Train_DARNN(model, data_prepared, 144, 'DARNN_EURUSD_H12.pt')