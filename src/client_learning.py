import random
import copy
import numpy as np
import torch as T
import torch.nn as nn
import math
import gc
from torch.utils.data import DataLoader
from typing import List, Optional, Union, Any, Dict
from logging import INFO
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_pinball_loss
from collections import OrderedDict
from src.utils.functions import inverse_transform_test, get_model
from src.utils.logger import log
from src.dataset.processing import Processing
from src.utils.early_stopping import EarlyStopping
from src.data import TimeSeriesLoader

class ClientLearning:
    def __init__(self, args, cid, seed: int):
        self.args = args
        self.cid = cid
        self.seed_all(seed)
        self.processing = Processing(args=self.args, data_path=self.args.data_path)
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.x_scaler = None
        self.y_scaler = None

        self.input_dim, self.output_dim = self.processing.get_data_shape()


        self.model = get_model(device=self.args.device, model=self.args.model_name, input_dim=self.input_dim,
                               out_dim=self.output_dim,
                               lags=self.args.num_lags)

    def _load_data(self):
        self.X_train, self.X_val, self.y_train, self.y_val, self.x_scaler, self.y_scaler = self.processing.make_preprocessing(filter_bs=self.cid, per_area=False, peek=False)

    def _unload_data(self):
        log(INFO, f"Client {self.cid}: Unloading data from RAM.")
        self.train_loader = None
        self.val_loader = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        gc.collect()

    def set_parameters(self, params: Union[List[np.ndarray], nn.Module]):
        if not isinstance(params, nn.Module):
            params_dict = zip(self.model.state_dict().keys(), params)
            state_dict = OrderedDict({k: T.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        else:
            self.model.load_state_dict(params.state_dict(), strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, params, criterion, optimizer, early_stopping, patience, lr, epochs, device):
        self.set_parameters(params)
        if self.train_loader is None or self.val_loader is None:
            self._load_data()
        self.model, train_loss_history, val_loss_history = self.train(model=self.model, epochs=epochs,
                                                            optimizer=optimizer, lr=lr, criterion=criterion,
                                                            early_stopping=early_stopping, patience=patience,
                                                            device=device)

        _, train_loss, train_metrics = self.evaluate(self.train_loader)
        num_val, val_loss, val_metrics = self.evaluate(self.val_loader)
        _train_instances = len(self.train_loader.dataset)
        self._unload_data()
        return self.get_parameters(), train_loss_history, _train_instances, train_loss, train_metrics, val_loss_history, num_val, val_loss, val_metrics


    def evaluate(self, data: Optional[Union[np.ndarray, DataLoader]]=None,
                 model: Optional[Union[nn.Module, List[np.ndarray]]]=None,
                 params: Optional[Dict[str, Any]]=None,
                 method: Optional[str]=None,
                 verbose: bool=False):

        if not params or "criterion" not in params:
            params = dict()
            params['criterion'] = nn.MSELoss()

        if model:
            self.set_parameters(model)



        if data is None and method == 'test':
            data = TimeSeriesLoader(X=self.X_val,
                         y=self.y_val,
                         num_lags=self.args.num_lags,
                         num_features=self.input_dim,
                         indices=[0], batch_size=self.args.batch_size, shuffle=False,
                         num_workers=self.args.num_workers).get_dataloader()
        if data is None and method == 'train':
            data = TimeSeriesLoader(X=self.X_train,
                         y=self.y_train,
                         num_lags=self.args.num_lags,
                         num_features=self.input_dim,
                         indices=[0], batch_size=self.args.batch_size, shuffle=False,
                         num_workers=self.args.num_workers).get_dataloader()

        loss, mse, rmse, mae, mape, r2, nrmse, pinball = self.test(self.model, data, params["criterion"], device=self.args.device)
        metrics = {"MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae), "MAPE": float(mape), 'R^2': float(r2), "pinball": float(pinball)}
        _instances = len(data.dataset)
        data = None
        gc.collect()
        return _instances, loss, metrics

    def test_model(self, params):
        self.set_parameters(params)
        self.X_test, self.y_test, num_features = self.processing.make_test_processing(filter_data=self.cid,
                                                                                      x_scaler=self.x_scaler,
                                                                                      y_scaler=self.y_scaler)

        test_loader = TimeSeriesLoader(X=self.X_test, y=self.y_test,
                                       num_lags=self.args.num_lags,
                                       num_features=num_features,
                                       indices=[0], batch_size=self.args.batch_size, shuffle=False,
                                       num_workers=self.args.num_workers).get_dataloader()

        test_mse, test_rmse, test_mae, test_mape, test_r2, test_nrmse, pinball, _, y_pred_test = self.test(self.model, test_loader,
                                                                                                    None,
                                                                                                    device=self.args.device)

        inverted_y_test, inverted_y_pred_test = inverse_transform_test(
            self.y_test, y_pred_test, self.y_scaler, round_preds=False, dims=[0]
        )

        inverted_test_mse, inverted_test_rmse, inverted_test_mae, inverted_test_mape, inverted_test_r2, inverted_test_nrmse, inverted_test_pinball, inverted_test_res_per_dim = self.accumulate_metrics(
            inverted_y_test, inverted_y_pred_test, log_per_output=True, return_all=True
            )

        results = {'y_true': self.y_test.tolist(), 'y_pred': y_pred_test.tolist(), 'mse': test_mse, 'rmse': test_rmse, 'mae': test_mae, 'mape': test_mape, 'r2': test_r2, 'nrmse': test_nrmse,
                   'pinball': pinball, 'client': self.cid}
        inverted_values = {'y_true': inverted_y_test.tolist(), 'y_pred': inverted_y_pred_test.tolist(), 'mse': inverted_test_mse, 'rmse': inverted_test_rmse, 'mae': inverted_test_mae, 'mape': inverted_test_mape,
                           'r2': inverted_test_r2, 'nrmse': inverted_test_nrmse, 'pinball': inverted_test_pinball,
                           'client': self.cid}

        return results, inverted_values



    def test(self, model: nn.Module, data, criterion, device: str="cuda"):
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
        mse, rmse, mae, mape, r2, nrmse, mean_pinball = self.accumulate_metrics(y_true.cpu(), y_pred.cpu())
        if criterion is None:
            return mse, rmse, mae, mape, r2, nrmse, mean_pinball, y_true.cpu(), y_pred.cpu()
        return loss, mse, rmse, mae, mape, r2, nrmse, mean_pinball


    def train(self, model: nn.Module, epochs: int=10, optimizer: str="adam",
              lr: float="1e-3", criterion: str="mse",
              early_stopping: bool=False, patience: int=50, device: str="cuda:0",
              log_per: int=1):

        best_model, best_loss, best_epoch = None, -1, -1
        train_loss_history, train_rmse_history = [], []
        val_loss_history, val_rmse_history, val_pinball_history = [], [], []
        if early_stopping:
            es_trace = True if log_per == 1 else False
            monitor = EarlyStopping(patience=patience, trace=es_trace)

        optimizer = self.get_optim(model=model, optim_name=optimizer, lr=lr)
        criterion = self.get_criterion(crit_name=criterion)

        for epoch in range(epochs):
            model.to(device)
            model.train()
            epochs_loss = []
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)

                loss.backward()

                optimizer.step()
                epochs_loss.append(loss.item())

            train_loss = sum(epochs_loss) / len(epochs_loss)
            _, train_mse, train_rmse, train_mae, train_mape, train_r2, train_nrmse, mean_pinball = self.test(model, self.train_loader,
                                                                              criterion, device)
            val_loss, val_mse, val_rmse, val_mae, val_mape, val_r2, val_nrmse, mean_pinball = self.test(model, self.val_loader,
                                                                                 criterion, device)
            log(INFO, f"Participant: {self.cid} | Epoch {epoch + 1}/{epochs} | [Train]: loss {train_loss:.6f}, MSE: {train_mse:.6f} | [Val]: loss {val_loss:.6f}, MSE: {val_mse:.6f}")
            train_loss_history.append(train_mse)
            train_rmse_history.append(train_rmse)
            val_loss_history.append(val_mse)
            val_rmse_history.append(val_rmse)
            val_pinball_history.append(mean_pinball)


            if early_stopping:
                monitor(val_loss, model)
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
                if best_loss == -1 or val_loss < best_loss:
                    best_loss = val_loss
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch + 1
        if early_stopping and epochs > patience:
            log(INFO, f"Participant: {self.cid} | Best loss: {best_loss}, Best Epoch: {best_epoch}")
        else:
            log(INFO, f"Participant: {self.cid} | Best loss: {best_loss}")
        return best_model, train_loss_history, val_loss_history

    def get_criterion(self, crit_name: str="mse"):
        if crit_name == "mse":
            return nn.MSELoss()
        elif crit_name == "l1":
            return nn.L1Loss()
        else:
            raise NotImplementedError(f"Criterion {crit_name} not supported")

    def get_optim(self, model: nn.Module, optim_name: str="adam", lr: float=1e-3):
        if optim_name == "adam":
            return T.optim.Adam(model.parameters(), lr=lr)
        elif optim_name == "sgd":
            return T.optim.SGD(model.parameters(), lr=lr)
        elif optim_name == "adamw":
            return T.optim.AdamW(model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Optimizer {optim_name} not supported")


    def log_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            shape = y_true.shape[1]
        except IndexError:
            return None
        assert y_true.shape == y_pred.shape


    def accumulate_metrics(self, y_true, y_pred, log_per_output: bool=False, dims: List[int]=[0], return_all: bool=False):
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
            res = self.log_metrics(y_true, y_pred)
            if return_all:
                return mse, rmse, mae, mape, r2, nrmse, mean_pinball, res
        return mse, rmse, mae, mape, r2, nrmse, mean_pinball


    def seed_all(self, seed: int = 0):
        random.seed(seed)
        np.random.seed(seed)
        T.manual_seed(seed)
        T.cuda.manual_seed_all(seed)
        T.use_deterministic_algorithms(True)
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False
