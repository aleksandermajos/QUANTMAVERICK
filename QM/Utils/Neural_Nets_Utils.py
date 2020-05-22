import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class InputAttentionEncoder(nn.Module):
    def __init__(self, N, M, T, stateful=False):
        """
        :param: N: int
            number of time serieses
        :param: M:
            number of LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with values of the last cell state
            of previous time window or to initialize it with zeros
        """
        super(self.__class__, self).__init__()
        self.N = N
        self.M = M
        self.T = T

        self.encoder_lstm = nn.LSTMCell(input_size=self.N, hidden_size=self.M)

        # equation 8 matrices

        self.W_e = nn.Linear( 2 *self.M, self.T)
        self.U_e = nn.Linear(self.T, self.T, bias=False)
        self.v_e = nn.Linear(self.T, 1, bias=False)

    def forward(self, inputs):
        encoded_inputs = torch.zeros((inputs.size(0), self.T, self.M)).cuda()

        # initiale hidden states
        h_tm1 = torch.zeros((inputs.size(0), self.M)).cuda()
        s_tm1 = torch.zeros((inputs.size(0), self.M)).cuda()

        for t in range(self.T):
            # concatenate hidden states
            h_c_concat = torch.cat((h_tm1, s_tm1), dim=1)

            # attention weights for each k in N (equation 8)
            x = self.W_e(h_c_concat).unsqueeze_(1).repeat(1, self.N, 1)
            y = self.U_e(inputs.permute(0, 2, 1))
            z = torch.tanh(x + y)
            e_k_t = torch.squeeze(self.v_e(z))

            # normalize attention weights (equation 9)
            alpha_k_t = F.softmax(e_k_t, dim=1)

            # weight inputs (equation 10)
            weighted_inputs = alpha_k_t * inputs[:, t, :]

            # calculate next hidden states (equation 11)
            h_tm1, s_tm1 = self.encoder_lstm(weighted_inputs, (h_tm1, s_tm1))

            encoded_inputs[:, t, :] = h_tm1
        return encoded_inputs


class TemporalAttentionDecoder(nn.Module):
    def __init__(self, M, P, T, stateful=False):
        """
        :param: M: int
            number of encoder LSTM units
        :param: P:
            number of deocder LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with values of the last cell state
            of previous time window or to initialize it with zeros
        """
        super(self.__class__, self).__init__()
        self.M = M
        self.P = P
        self.T = T
        self.stateful = stateful

        self.decoder_lstm = nn.LSTMCell(input_size=1, hidden_size=self.P)

        # equation 12 matrices
        self.W_d = nn.Linear(2 * self.P, self.M)
        self.U_d = nn.Linear(self.M, self.M, bias=False)
        self.v_d = nn.Linear(self.M, 1, bias=False)

        # equation 15 matrix
        self.w_tilda = nn.Linear(self.M + 1, 1)

        # equation 22 matrices
        self.W_y = nn.Linear(self.P + self.M, self.P)
        self.v_y = nn.Linear(self.P, 1)

    def forward(self, encoded_inputs, y):
        # initializing hidden states
        d_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).cuda()
        s_prime_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).cuda()
        for t in range(self.T):
            # concatenate hidden states
            d_s_prime_concat = torch.cat((d_tm1, s_prime_tm1), dim=1)
            # print(d_s_prime_concat)
            # temporal attention weights (equation 12)
            x1 = self.W_d(d_s_prime_concat).unsqueeze_(1).repeat(1, encoded_inputs.shape[1], 1)
            y1 = self.U_d(encoded_inputs)
            z1 = torch.tanh(x1 + y1)
            l_i_t = self.v_d(z1)

            # normalized attention weights (equation 13)
            beta_i_t = F.softmax(l_i_t, dim=1)

            # create context vector (equation_14)
            c_t = torch.sum(beta_i_t * encoded_inputs, dim=1)

            # concatenate c_t and y_t
            y_c_concat = torch.cat((c_t, y[:, t, :]), dim=1)
            # create y_tilda
            y_tilda_t = self.w_tilda(y_c_concat)

            # calculate next hidden states (equation 16)
            d_tm1, s_prime_tm1 = self.decoder_lstm(y_tilda_t, (d_tm1, s_prime_tm1))

        # concatenate context vector at step T and hidden state at step T
        d_c_concat = torch.cat((d_tm1, c_t), dim=1)

        # calculate output
        y_Tp1 = self.v_y(self.W_y(d_c_concat))
        return y_Tp1

class DARNN(nn.Module):
    def __init__(self, N, M, P, T, stateful_encoder=False, stateful_decoder=False):
        super(self.__class__, self).__init__()
        self.encoder = InputAttentionEncoder(N, M, T, stateful_encoder).cuda()
        self.decoder = TemporalAttentionDecoder(M, P, T, stateful_decoder).cuda()
    def forward(self, X_history, y_history):
        out = self.decoder(self.encoder(X_history), y_history)
        return out

def TrainDARNN(name,model,epochs,loss,opt,epoch_scheduler,data_train_loader,data_val_loader):
    patience = 50
    min_val_loss = 9999
    counter = 0
    for i in range(epochs):
        mse_train = 0
        for batch_x, batch_y_h, batch_y in data_train_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_y_h = batch_y_h.cuda()
            opt.zero_grad()
            y_pred = model(batch_x, batch_y_h)
            y_pred = y_pred.squeeze(1)
            l = loss(y_pred, batch_y)
            l.backward()
            mse_train += l.item() * batch_x.shape[0]
            opt.step()
        epoch_scheduler.step()
        with torch.no_grad():
            mse_val = 0
            preds = []
            true = []
            for batch_x, batch_y_h, batch_y in data_val_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_y_h = batch_y_h.cuda()
                output = model(batch_x, batch_y_h)
                output = output.squeeze(1)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())
                mse_val += loss(output, batch_y).item() * batch_x.shape[0]
        preds = np.concatenate(preds)
        true = np.concatenate(true)
        print(i)

        if min_val_loss > mse_val ** 0.5:
            min_val_loss = mse_val ** 0.5
            print("Saving...")
            torch.save(model.state_dict(), name+".pt")
            counter = 0
        else:
            counter += 1

        if counter == patience:
            break
    return model

class Predictor():
    def __init__(self):
        self.loss = nn.MSELoss()
        self.model = DARNN(3, 64, 64, 16).cuda()
        self.model.load_state_dict(torch.load("darnn_OANDAMT4LIVE_EURUSD240.pt"))

        data = pd.read_csv("./Data/EURUSD240.csv")
        to_remove = ["T1", "T2", 'V']
        data = data[data.columns.difference(to_remove)]

        batch_size = 16
        timesteps = 16
        train_length = data.shape[0]-116
        val_length = 100
        test_length = 16

        X = np.zeros((len(data), timesteps, data.shape[1] - 1))
        Y = np.zeros((len(data), timesteps, 1))

        for i, name in enumerate(list(data.columns[:-1])):
            for j in range(timesteps):
                X[:, j, i] = data[name].shift(timesteps - j - 1).fillna(method="bfill")

        for j in range(timesteps):
            Y[:, j, 0] = data["O"].shift(timesteps - j - 1).fillna(method="bfill")

        prediction_horizon = 1
        target = data["O"].shift(-prediction_horizon).fillna(method="ffill").values

        X = X[timesteps:]
        Y = Y[timesteps:]
        target = target[timesteps:]

        X_train = X[:train_length]
        X_val = X[train_length:train_length + val_length]
        X_test = X[train_length + val_length - timesteps:train_length + val_length + test_length - timesteps]
        Y_his_train = Y[:train_length]
        Y_his_val = Y[train_length:train_length + val_length]
        Y_his_test = Y[train_length + val_length - timesteps:train_length + val_length + test_length - timesteps]
        target_train = target[:train_length]
        target_val = target[train_length:train_length + val_length]
        target_test = target[train_length + val_length - timesteps:train_length + val_length + test_length - timesteps]

        X_train_max = X_train.max(axis=0)
        X_train_min = X_train.min(axis=0)
        Y_his_train_max = Y_his_train.max(axis=0)
        Y_his_train_min = Y_his_train.min(axis=0)
        self.target_train_max = target_train.max(axis=0)
        self.target_train_min = target_train.min(axis=0)

        X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
        X_val = (X_val - X_train_min) / (X_train_max - X_train_min)
        X_test = (X_test - X_train_min) / (X_train_max - X_train_min)
        Y_his_train = (Y_his_train - Y_his_train_min) / (Y_his_train_max - Y_his_train_min)
        Y_his_val = (Y_his_val - Y_his_train_min) / (Y_his_train_max - Y_his_train_min)
        Y_his_test = (Y_his_test - Y_his_train_min) / (Y_his_train_max - Y_his_train_min)
        target_train = (target_train - self.target_train_min) / (self.target_train_max - self.target_train_min)
        target_val = (target_val - self.target_train_min) / (self.target_train_max - self.target_train_min)
        target_test = (target_test - self.target_train_min) / (self.target_train_max - self.target_train_min)

        X_train_t = torch.Tensor(X_train)
        X_val_t = torch.Tensor(X_val)
        X_test_t = torch.Tensor(X_test)
        Y_his_train_t = torch.Tensor(Y_his_train)
        Y_his_val_t = torch.Tensor(Y_his_val)
        Y_his_test_t = torch.Tensor(Y_his_test)
        target_train_t = torch.Tensor(target_train)
        target_val_t = torch.Tensor(target_val)
        target_test_t = torch.Tensor(target_test)

        from torch.utils.data import TensorDataset, DataLoader
        data_train_loader = DataLoader(TensorDataset(X_train_t, Y_his_train_t, target_train_t), shuffle=True,
                                       batch_size=batch_size)
        data_val_loader = DataLoader(TensorDataset(X_val_t, Y_his_val_t, target_val_t), shuffle=False,
                                     batch_size=batch_size)
        self.data_test_loader = DataLoader(TensorDataset(X_test_t, Y_his_test_t, target_test_t), shuffle=False,
                                      batch_size=batch_size)

    def MakePredictions(self):
        with torch.no_grad():
            mse_val = 0
            preds = []
            true = []
            for batch_x, batch_y_h, batch_y in self.data_test_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_y_h = batch_y_h.cuda()
                output = self.model(batch_x, batch_y_h)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())
                mse_val += self.loss(output, batch_y).item() * batch_x.shape[0]
        preds = np.concatenate(preds)
        true = np.concatenate(true)

        preds = preds * (self.target_train_max - self.target_train_min) + self.target_train_min
        true = true * (self.target_train_max - self.target_train_min) + self.target_train_min

        right = 0
        wrong = 0
        for i in range(len(preds) - 1):
            if preds[i + 1][0] - preds[i][0] > 0 and true[i + 1] - true[i] < 0 or preds[i + 1][0] - preds[i][0] < 0 and \
                    true[i + 1] - true[i] > 0:
                wrong += 1
            if preds[i + 1][0] - preds[i][0] < 0 and true[i + 1] - true[i] > 0 or preds[i + 1][0] - preds[i][0] > 0 and \
                    true[
                        i + 1] - true[i] < 0:
                wrong += 1
            if preds[i + 1][0] - preds[i][0] > 0 and true[i + 1] - true[i] > 0 or preds[i + 1][0] - preds[i][0] < 0 and \
                    true[i + 1] - true[i] < 0:
                right += 1
        wrong = len(preds) - right
        precentage_right = right / (right + wrong) * 100
        print(precentage_right)

        return preds[len(preds)-1]

