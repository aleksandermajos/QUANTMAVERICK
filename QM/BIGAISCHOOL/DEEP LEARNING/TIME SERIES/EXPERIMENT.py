from QM.DATA.DataGenerator import OandaDataGenerator
dg = OandaDataGenerator()
from QM.DATA.DataPreparator import PyTorchDataPreparator
import sys, os
parent_dir = os.getcwd()
path = os.path.dirname(parent_dir)
sys.path.append(path)
sys.path.append('C:\\Users\\hapir\\PycharmProjects\\quantmaverick\\QM\\BIGAISCHOOL\\DEEP LEARNING\\ARCHITECTURES')
sys.path.append('C:\\Users\\hapir\\PycharmProjects\\quantmaverick\\QM\\BIGAISCHOOL\\DEEP LEARNING\\TRAINING')
from DARNN import DARNN
from Train import Train_DARNN


data = dg.GetData("2018-01-01T00:00:00Z","2020-02-14T00:00:00Z","EUR_USD","H12","DF")
to_remove = ["time","volume"]
data = data[data.columns.difference(to_remove)]
pdp = PyTorchDataPreparator(data)
data_prepared = pdp.GetDataPrepared(128,16,80,10,10,"O")
model = DARNN(data_prepared['X_train'].shape[2], 64, 64, data_prepared['X_train'].shape[1]).cuda()
Train_DARNN(model, data_prepared, 144, 'DARNN_EURUSD_H12.pt')







