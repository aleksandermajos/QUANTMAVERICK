from Utils.Data_Utils_Preparator import df_to_tensor
from torch.utils.data import TensorDataset, DataLoader


class PyTorchDataPreparator():
    def __init__(self,data):
        self.data = data

    def GetDataPrepared(self, batch_size, time_steps, train_perc, val_perc, test_perc, target):
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.train_perc = train_perc
        self.val_perc = val_perc
        self.test_perc = test_perc
        self.target = target
        X_train, target_train_max, target_train_min, X_train_t, X_val_t, X_test_t, y_his_train_t, y_his_val_t, y_his_test_t, \
        target_train_t, target_val_t, target_test_t = df_to_tensor(self.data, self.batch_size, self.time_steps, self.train_perc, self.val_perc, self.test_perc , self.target)

        data_train_loader = DataLoader(TensorDataset(X_train_t, y_his_train_t, target_train_t), shuffle=True,
                                       batch_size=self.batch_size)
        data_val_loader = DataLoader(TensorDataset(X_val_t, y_his_val_t, target_val_t), shuffle=False, batch_size=self.batch_size)
        data_test_loader = DataLoader(TensorDataset(X_test_t, y_his_test_t, target_test_t), shuffle=False,
                                      batch_size=self.batch_size)

        Results = {'X_train': X_train, 'target_train_max': target_train_max, 'target_train_min': target_train_min,
                   'X_train_t': X_train_t,'X_val_t': X_val_t,'X_test_t': X_test_t,'y_his_train_t': y_his_train_t,
                   'y_his_val_t': y_his_val_t,'y_his_test_t': y_his_test_t,'target_train_t': target_train_t,
                   'target_val_t': target_val_t,'target_test_t': target_test_t,
                   'data_train_loader': data_train_loader,'data_val_loader':data_val_loader,'data_test_loader':data_test_loader}
        return Results



