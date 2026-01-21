
import copy
import math
import numpy as np

import torch as T
import torch.nn as nn
import random
from logging import INFO
from src.utils.logger import log
# from carbontracker.tracker import CarbonTracker
from torch.utils.data import DataLoader
from src.utils.early_stopping import EarlyStopping
from src.models.cnn import CNN
from src.models.gru import GRU
from src.models.lstm import LSTM
from src.models.rnn import RNN
from tqdm import tqdm
from typing import Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_pinball_loss, mean_absolute_percentage_error
from src.utils.functions import *




class Trainers:
    def __init__(self, args):
        self.args = args


    def fit(self):
        raise NotImplemented


    @staticmethod
    def seed_all(seed: int = 0):
        random.seed(seed)
        np.random.seed(seed)
        T.manual_seed(seed)
        T.cuda.manual_seed_all(seed)
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = True

    @staticmethod
    def get_optim(model: nn.Module, optim_name: str="adam", lr: float=1e-3):
        if optim_name == "adam":
            return T.optim.Adam(model.parameters(), lr=lr)
        elif optim_name == "sgd":
            return T.optim.SGD(model.parameters(), lr=lr)
        elif optim_name == "adamw":
            return T.optim.AdamW(model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Optimizer {optim_name} not supported")


    @staticmethod
    def get_criterion(crit_name: str="mse"):
        if crit_name == "mse":
            return nn.MSELoss()
        elif crit_name == "l1":
            return nn.L1Loss()
        else:
            raise NotImplementedError(f"Criterion {crit_name} not supported")

    @staticmethod
    def log_metrics(y_true: np.ndarray, y_pred: np.ndarray):
        try:
            shape = y_true.shape[1]
        except IndexError:
            return None
        assert y_true.shape == y_pred.shape

    @classmethod
    def accumulate_metrics(cls, y_true, y_pred, log_per_output: bool=False, dims: List[int]=[0], return_all: bool=False):
        if not isinstance(y_true, np.ndarray):
            y_true = y_true.cpu().numpy()
        if not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.cpu().numpy()

        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mean_pinball = mean_pinball_loss(y_true, y_pred, alpha=1)


        y_true_first_dim = y_true[:, dims[0]]
        y_pred_first_dim = y_pred[:, dims[0]]

        rmse_first_dim = math.sqrt(mean_squared_error(y_true_first_dim, y_pred_first_dim))
        nrmse_first_dim = rmse_first_dim/np.mean(y_true_first_dim)

        if y_true.shape[1] >= 2:
            nrmses = 0
            for i in range(1, len(dims)):
                y_true_dim = y_true[:, dims[i]]
                y_pred_dim = y_pred[:, dims[i]]
                rmse_dim = math.sqrt(mean_squared_error(y_true_dim, y_pred_dim))
                nrmse_dim = rmse_dim / np.mean(y_true_dim)
                nrmses += nrmse_dim
            nrmse = (nrmse_first_dim + nrmses) / len(dims)
        else:
            nrmse = nrmse_first_dim

        if log_per_output:
            res = cls.log_metrics(y_true, y_pred)
            if return_all:
                return mse, rmse, mae, mape, r2, nrmse, mean_pinball, res
        return mse, rmse, mae, mape, r2, nrmse, mean_pinball



    @classmethod
    def test(cls, model: nn.Module, data, criterion, device: str="cuda"):
        model.to(device)
        model.eval()
        y_true, y_pred = [], []
        loss = 0.0
        with T.no_grad():
            for x, y in data:
                x, y = x.to(device), y.to(device)
                out = model(x)
                if criterion is not None:
                    loss += criterion(out, y).item()
                y_true.extend(y)
                y_pred.extend(out)
        loss /= len(data.dataset)

        y_true = T.stack(y_true)
        y_pred = T.stack(y_pred)
        mse, rmse, mae, mape, r2, nrmse, mean_pinball = cls.accumulate_metrics(y_true.cpu(), y_pred.cpu())
        if criterion is None:
            return mse, rmse, mae, mape, r2, nrmse, mean_pinball, y_true.cpu(), y_pred.cpu()
        return loss, mse, rmse, mae, mape, r2, nrmse, mean_pinball

    @classmethod
    def train(cls, model: nn.Module, cid: str, train_loader: DataLoader, test_loader: DataLoader, epochs: int=10, optimizer: str="adam",
              lr: float="1e-3", reg1: float=0, reg2: float=0, max_grad_norm: float=0, criterion: str="mse",
              early_stopping: bool=False, patience: int=50, plot_history: bool=False, device: str="cuda:0", fedprox_mu: float=0.0,
              log_per: int=1, use_carbontracker: bool=False, fl_round=0, model_name: Optional[str]=''):
        # wandb_logger = wandb.init(project='FL-ConsumptionForecasting-Scratch',
        #                                tags=['centralized', 'consumption'], group='Centralized', name=f"FL_Round_{fl_round}")

        best_model, best_loss, best_epoch = None, -1, -1
        train_loss_history, train_rmse_history = [], []
        test_loss_history, test_rmse_history, test_pinball_history = [], [], []
        if early_stopping:
            es_trace = True if log_per == 1 else False
            monitor = EarlyStopping(patience=patience, trace=es_trace)

        optimizer = cls.get_optim(model=model, optim_name=optimizer, lr=lr)
        criterion = cls.get_criterion(crit_name=criterion)
        global_weight_collector = copy.deepcopy(list(model.parameters()))

        for epoch in range(epochs):
            model.to(device)
            model.train()
            epochs_loss = []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)

                # if fedprox_mu > 0.0:
                #     fedprox_reg = 0.0
                #     for param_index, param in enumerate(model.parameters()):
                #         fedprox_reg += ( (fedprox_mu / 2) * T.norm((param - global_weight_collector[param_index])) ** 2 )
                #     loss += fedprox_reg
                # if reg1 > 0.0:
                #     params = T.cat([p.view(-1) for name, p in model.named_parameters() if "bias" not in name])
                #     loss += reg1 * T.norm(params, 1)
                # if reg2 > 0.0:
                #     params = T.cat([p.view(-1) for name, p in model.named_parameters() if "bias" not in name])
                #     loss += reg2 * T.norm(params, 2)
                loss.backward()

                # if max_grad_norm > 0.0:
                #     nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                epochs_loss.append(loss.item())

            train_loss = sum(epochs_loss) / len(epochs_loss)
            _, train_mse, train_rmse, train_mae, train_mape, train_r2, train_nrmse, mean_pinball = cls.test(model, train_loader,
                                                                              criterion, device)
            test_loss, test_mse, test_rmse, test_mae, test_mape, test_r2, test_nrmse, mean_pinball = cls.test(model, test_loader,
                                                                                 criterion, device)
            log(INFO, f"Participant: {cid} | Epoch {epoch + 1}/{epochs} | [Train]: loss {train_loss}, mse: {train_mse} | [Test]: loss {test_loss}, mse: {test_mse}")
            # wandb_logger.log({'cid': cid, 'epoch': epoch + 1, 'train_loss': train_loss, 'train_mse': train_mse, 'test_loss': test_loss, 'test_mse': test_mse})
            train_loss_history.append(train_mse)
            train_rmse_history.append(train_rmse)
            test_loss_history.append(test_mse)
            test_rmse_history.append(test_rmse)
            test_pinball_history.append(mean_pinball)


            if early_stopping:
                monitor(test_loss, model)
                best_loss = abs(monitor.best_score)
                best_model = monitor.best_model
                if epoch + 1 > patience:
                    best_epoch = epochs + 1
                elif epoch + 1 == epochs:
                    best_epoch = epochs + 1 - monitor.counter
                else:
                    best_epoch = epoch + 1 - patience
                if monitor.early_stop:
                    log(INFO, "Early Stopping")
                    break
            else:
                if best_loss == -1 or test_loss < best_loss:
                    best_loss = test_loss
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch + 1
        if plot_history:
            plot_train_curve(train_loss_history, test_loss_history, "MSE curve", 'MSE_curve', model_name)
            plot_train_curve(train_rmse_history, test_rmse_history, "RMSE curve", 'RMSE_curve',model_name)
        if early_stopping and epochs > patience:
            log(INFO, f"Participant: {cid} | Best loss: {best_loss}, Best Epoch: {best_epoch}")
        else:
            log(INFO, f"Participant: {cid} | Best loss: {best_loss}")
        return best_model, train_loss_history, test_loss_history


