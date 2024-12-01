from collections import OrderedDict

import torch
import pickle
from torch import nn
from torch import optim
from ltv_system import LTVSystem, LTVParameter
import numpy as np


class PredictionNetwork(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 20)
        )
        # Q, R, x_bar, A, B, w, Q_terminal, x_bar_terminal
        # dimension: 2, 2, 2, 4, 4, 2, 2, 2

    def forward(self, x):
        out = self.model(x)
        # Q, R, x_bar, A, B, w, Q_terminal, x_bar_terminal
        # 2, 2, 2, 4, 4, 2, 2, 2
        # out[:,0:4] = torch.clamp(out[:, 0:4].clone(), min=1e-6)
        return out


class LTVSystemWithNeuralNetPrediction(LTVSystem):
    def __init__(self, prediction_nn, **kwargs):
        super().__init__(**kwargs)
        self.prediction_nn = prediction_nn

    def get_program_parameters(self, time_step):
        # Q, R, x_bar, A, B, w, Q_terminal, x_bar_terminal
        out = self.prediction_nn(torch.tensor([time_step], dtype=torch.float32)).cpu().detach()
        Q = np.eye(2)
        Q[[0, 1],[0, 1]] = np.clip(out[0:2], a_min=1e-6, a_max=None)
        R = np.eye(2)
        R[[0, 1], [0, 1]] = np.clip(out[2:4], a_min=1e-6, a_max=None)
        x_bar = out[4:6]
        A = out[6:10].reshape(2,2)
        B = out[10:14].reshape(2,2)
        w = out[14:16]
        Q_terminal = np.eye(2)
        Q_terminal[[0, 1], [0, 1]] = np.clip(out[16:18], a_min=1e-6, a_max=None)
        x_bar_terminal = out[18:20]
        return LTVParameter(Q, R, x_bar, A, B, w, Q_terminal, x_bar_terminal)


def read_reference_into_pytorch(fn, dt):
    results = pickle.load(open(fn, "rb"))
    parameters = results.opt_datas.parameters
    out = []
    for p in parameters:
        o = np.zeros(20, dtype=np.float32)
        o[:2] = p.Q[[0, 1],[0, 1]]
        o[2:4] = p.R[[0, 1], [0, 1]]
        o[4:6] = p.x_bar
        o[6:10] = p.A.flatten()
        o[10:14] = p.B.flatten()
        o[14:16] = p.w
        o[16:18] = p.Q_terminal[[0, 1], [0, 1]]
        o[18:20] = p.x_bar_terminal
        out.append(o)
    in_tensor = torch.arange(len(parameters))[:, None] * dt
    out_tensor = torch.from_numpy(np.stack(out))
    return in_tensor, out_tensor


class NNPredTrainer:
    def __init__(self, fn, dt):
        self.in_tensor, self.out_tensor = read_reference_into_pytorch(fn, dt)
        self.pnn = PredictionNetwork()
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.pnn.parameters(), lr=1e-4)
        self.eval_data = []

    def train(self, num_iterations, verbose=1000):
        for j in range(num_iterations):
            self.optimizer.zero_grad()
            p = self.pnn(self.in_tensor)
            loss = self.loss_fn(p, self.out_tensor)
            loss.backward()
            self.optimizer.step()
            l = loss.cpu().detach().item()
            self.eval_data.append(self.evaluation(p.detach().cpu(), self.out_tensor.detach().cpu()))
            if verbose is not None and (verbose == 0 or (len(self.eval_data)-1) % verbose == 0):
                print(f"Iteration: {len(self.eval_data)-1}: loss={self.eval_data[-1]}")

    def evaluation(self, prediction, target):
        mse = ((prediction - target) ** 2).mean()
        mae = np.abs(prediction - target).mean()
        return OrderedDict(MSE=mse, MAE=mae)


if __name__ == '__main__':
    trainer = NNPredTrainer("data/offline/ltv/seed-100", 0.1)
    trainer.train(10000)