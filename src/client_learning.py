import os
import pickle
import random
import copy
import numpy as np
import torch as T
import torch.nn as nn
import math
import gc
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List, Optional, Union, Any, Dict
from logging import INFO, DEBUG
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_pinball_loss
from collections import OrderedDict, defaultdict
from src.models.timeVAE.timevae import TimeVAE
from src.utils.functions import inverse_transform_test, mkdir_if_not_exists, seed_all
from src.utils.logger import log
from src.data import LocalFileDataset
from src.utils.early_stopping import EarlyStopping

class ClientLearning:
    def __init__(self, args, cid, seed: int, hparams: Optional[dict]=None):
        self.args = args
        self.cid = cid
        with open(f"{self.args.data_path}/{self.args.filter_bs}_metadata.pkl", "rb") as f:
            _meta_doc =  pickle.load(f)

        self.input_dim = _meta_doc["input_dim"]
        self.output_dim = _meta_doc["output_dim"]
        self.x_scaler = pickle.loads(_meta_doc["x_scaler"])
        self.y_scaler = pickle.loads(_meta_doc["y_scaler"])
        self.train_dataset = LocalFileDataset(client_id=self.args.filter_bs, _type="train", data_path=self.args.data_path)
        self.val_dataset = LocalFileDataset(client_id=self.args.filter_bs, _type="val", data_path=self.args.data_path)

        # if wandb.run is None:
        #     wandb.init(
        #         project=getattr(self.args, 'wandb_project', 'fl_simulation'),
        #         group=getattr(self.args, 'wandb_group', 'experiment_1'),
        #         name=f"client_{self.cid}",
        #         job_type="client_train",
        #         config=vars(self.args),
        #         reinit=True
        #     )

        if hparams:
            for k, v in hparams.items():
                setattr(self.args, k, v)
            # wandb.config.update(hparams, allow_val_change=True)

        self.model = None

    def get_latent_space(self, latent_dim: int=8, epochs: int=8):
        from src.models.timeVAE.timevae import TimeVAE
        timevae = TimeVAE(hidden_sizes=self.args.hidden_dims, trend_poly=self.args.trend_poly,
                          custom_seats=self.args.custom_seats,
                          use_residual_conn=self.args.use_residual_conn, seq_len=self.args.num_lags,
                          feat_dim=self.input_dim, latent_dim=latent_dim, device=self.args.device)

        train_loader, val_loader = self._load_data(shuffle=True)

        mkdir_if_not_exists(f'etc/TimeVAE/{self.args.loc}/ckpt/')
        mkdir_if_not_exists(f'etc/TimeVAE/{self.args.loc}/logs/')

        if os.path.exists(f'etc/TimeVAE/{self.args.loc}/ckpt/{self.cid}-latent_dim_{latent_dim}.pth'):
            log(INFO, f"{self.cid}'s TimeVAE model found. Loading state dict")
            timevae.load_state_dict(T.load(f'etc/TimeVAE/{self.args.loc}/ckpt/{self.cid}-latent_dim_{latent_dim}.pth', map_location=self.args.device))
        else:
            log(INFO, f"{self.cid}'s TimeVAE model for latent dim {latent_dim} not found. Training client's model")
            timevae, train_val_log = self.fit_timevae(timevae, train_loader, val_loader, epochs)
            T.save(timevae.state_dict(), f'etc/TimeVAE/{self.args.loc}/ckpt/{self.cid}-latent_dim_{latent_dim}.pth')

            with open(f'etc/TimeVAE/{self.args.loc}/logs/{self.cid}-latent_dim-{latent_dim}-train_val.pkl', "wb") as f:
                pickle.dump(train_val_log, f)
                log(INFO, f"TimeVAE training log saved on etc/TimeVAE/{self.args.loc}/logs/{self.cid}-latent_dim-{latent_dim}-train_val.pkl")

        timevae.eval()
        latent_vectors = []

        with T.no_grad():
            for X, _ in train_loader:
                X = X.to(self.args.device)
                X = X.reshape(X.size(0), X.size(1), X.size(2))

                z_mean, _, _ = timevae.encoder(X)
                latent_vectors.append(z_mean.cpu())

        all_latents = T.cat(latent_vectors, dim=0)
        client_signature = all_latents.mean(dim=0).numpy()

        return client_signature

    def fit_timevae(self, timevae: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int=8):
        timevae, train_val_log = self.train_timevae(timevae, train_loader, val_loader, epochs)
        return timevae, train_val_log

    def _get_reconstruction_loss(self, X, X_recons):
        def get_reconst_loss_by_axis(X, X_recons, dim):
            x_r = T.mean(X, dim=dim)
            x_c_r = T.mean(X_recons, dim=dim)
            err = T.pow(x_r - x_c_r, 2)
            loss = T.sum(err)
            return loss

        err = T.pow(X - X_recons, 2)
        reconst_loss = T.sum(err)

        reconst_loss += get_reconst_loss_by_axis(X, X_recons, dim=2)  # by time axis
        reconst_loss += get_reconst_loss_by_axis(X, X_recons, dim=1)  # by feature axis

        return reconst_loss

    def loss_function(self, X, X_recons, z_mean, z_log_var):
        reconstruction_loss = self._get_reconstruction_loss(X, X_recons)
        kl_loss = -0.5 * T.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        total_loss = self.args.reconstruction_wt * reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss


    def train_timevae(self, model: nn.Module, train_loader, val_loader, max_epochs):
        timevae_loss_dict = defaultdict(list)
        optimizer = T.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
        best_val_loss = np.inf
        best_model = None
        with tqdm(total=max_epochs, desc=f"Training {self.cid}'s TimeVAE model") as pbar:
            model.train()
            model.to(self.args.device)
            total_loss = []
            reconstruction_loss = []
            kl_loss = []

            for X, y in train_loader:
                X, y = X.to(self.args.device), y.to(self.args.device)
                X = X.reshape(X.size(0), X.size(1), X.size(2))
                optimizer.zero_grad()
                z_mean, z_log_var, z = model.encoder(X)
                reconstruction = model.decoder(z)
                loss, recon_loss, kl = self.loss_function(X, reconstruction, z_mean, z_log_var)
                loss = loss / X.size(0)
                recon_loss = recon_loss / X.size(0)
                kl = kl / X.size(0)

                loss.backward()
                optimizer.step()

                total_loss.append(loss.item())
                reconstruction_loss.append(recon_loss.item())
                kl_loss.append(kl.item())

            val_loss, val_reconstruction_loss, val_kl_loss = self.test_timevae(model, val_loader)
            timevae_loss_dict["train_total_loss"].append(np.mean(total_loss))
            timevae_loss_dict["train_reconstruction_loss"].append(np.mean(reconstruction_loss))
            timevae_loss_dict["train_kl_loss"].append(np.mean(kl_loss))
            timevae_loss_dict["val_total_loss"].append(val_loss)
            timevae_loss_dict["val_reconstruction_loss"].append(val_reconstruction_loss)
            timevae_loss_dict["val_kl_loss"].append(val_kl_loss)

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
            pbar.set_postfix({'loss': round(val_loss, 4),
                              'recon_loss': round(val_reconstruction_loss, 4),
                              'kl_loss': round(val_kl_loss, 4)})
            pbar.update()
        return best_model, timevae_loss_dict

    def test_timevae(self, model, val_loader):
        model.to(self.args.device)
        model.eval()

        total_loss = []
        reconstruction_loss = []
        kl_loss = []

        with T.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.args.device), y.to(self.args.device)
                X = X.reshape(X.size(0), X.size(1), X.size(2))
                z_mean, z_log_var, z = model.encoder(X)
                reconstruction = model.decoder(z)
                loss, recon_loss, kl = self.loss_function(X, reconstruction, z_mean, z_log_var)
                loss = loss / X.size(0)
                recon_loss = recon_loss / X.size(0)
                kl = kl / X.size(0)
                total_loss.append(loss.item())
                reconstruction_loss.append(recon_loss.item())
                kl_loss.append(kl.item())
        return np.mean(total_loss), np.mean(reconstruction_loss), np.mean(kl_loss)

    def prepare_model(self, params=None):
        from src.utils.functions import get_model
        self.model = get_model(device=self.args.device, model=self.args.model_name, input_dim=self.input_dim,
                               out_dim=self.output_dim,
                               lags=self.args.num_lags)
        if params:
            self.set_parameters(params)

    def clean_up(self):
        """Limpa o modelo da VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if T.cuda.is_available():
            T.cuda.empty_cache()
        gc.collect()

    def _load_data(self, shuffle: bool=False):
        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=shuffle, num_workers=self.args.num_workers, pin_memory=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=shuffle, num_workers=self.args.num_workers, pin_memory=True)
        return train_loader, val_loader


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
        self.prepare_model(params)
        log(DEBUG, f"Client {self.cid} waiting GPU gueue")
        train_loader, val_loader = self._load_data()
        self.model, train_loss_history, val_loss_history = self.train(train_loader=train_loader, val_loader=val_loader, model=self.model, epochs=epochs,
                                                            optimizer=optimizer, lr=lr, criterion=criterion,
                                                            early_stopping=early_stopping, patience=patience,
                                                            device=device)

        _, train_loss, train_metrics = self.evaluate(train_loader)
        num_val, val_loss, val_metrics = self.evaluate(val_loader)
        _train_instances = len(self.train_dataset)
        del train_loader
        del val_loader
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
            self.prepare_model(model)

        if data is None and method == 'test':
            data = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        if data is None and method == 'train':
            data = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

        loss, mse, rmse, mae, mape, r2, nrmse, pinball, _, _ = self.test(self.model, data, params["criterion"], device=self.args.device)
        metrics = {"MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae), "MAPE": float(mape), 'R^2': float(r2), "pinball": float(pinball)}
        _instances = len(data.dataset)
        del data
        return _instances, loss, metrics

    def test_model(self, params):
        self.prepare_model(params)

        test_dataset = LocalFileDataset(client_id=self.args.filter_bs, _type="test", data_path=self.args.test_path)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=self.args.num_workers)

        test_mse, test_rmse, test_mae, test_mape, test_r2, test_nrmse, pinball, _, y_pred_test = self.test(self.model, test_loader,
                                                                                                    None,
                                                                                                    device=self.args.device)

        y_test = test_dataset.y
        inverted_y_test, inverted_y_pred_test = inverse_transform_test(
            y_test, y_pred_test, self.y_scaler, round_preds=False, dims=[0]
        )

        inverted_test_mse, inverted_test_rmse, inverted_test_mae, inverted_test_mape, inverted_test_r2, inverted_test_nrmse, inverted_test_pinball, inverted_test_res_per_dim = self.accumulate_metrics(
            inverted_y_test, inverted_y_pred_test, log_per_output=True, return_all=True
            )

        results = {'y_true': y_test.tolist(), 'y_pred': y_pred_test.tolist(), 'mse': test_mse, 'rmse': test_rmse, 'mae': test_mae, 'mape': test_mape, 'r2': test_r2, 'nrmse': test_nrmse,
                   'pinball': pinball, 'client': self.cid}
        inverted_values = {'y_true': inverted_y_test.tolist(), 'y_pred': inverted_y_pred_test.tolist(), 'mse': inverted_test_mse, 'rmse': inverted_test_rmse, 'mae': inverted_test_mae, 'mape': inverted_test_mape,
                           'r2': inverted_test_r2, 'nrmse': inverted_test_nrmse, 'pinball': inverted_test_pinball,
                           'client': self.cid}

        return results, inverted_values



    def test(self, model: nn.Module, data, criterion, device: str="cuda"):
        model.to(device)
        model.eval()
        y_true, y_pred = [], []
        loss_sum = T.tensor(0.0, device=device)
        with T.no_grad():
            for x, y in data:
                x, y = x.to(device), y.to(device)
                out = model(x)
                if criterion is not None:
                    loss_sum += criterion(out, y).detach()
                y_true.append(y)
                y_pred.append(out)
        loss = (loss_sum / len(data.dataset)).item()

        y_true = T.cat(y_true, dim=0)
        y_pred = T.cat(y_pred, dim=0)
        mse, rmse, mae, mape, r2, nrmse, mean_pinball = self.accumulate_metrics(y_true.cpu(), y_pred.cpu())
        del model
        del data
        if criterion is None:
            return mse, rmse, mae, mape, r2, nrmse, mean_pinball, y_true.cpu(), y_pred.cpu()
        return loss, mse, rmse, mae, mape, r2, nrmse, mean_pinball, y_true.cpu(), y_pred.cpu()


    def train(self, train_loader: DataLoader, val_loader: DataLoader, model: nn.Module, epochs: int=10, optimizer: str="adam",
              lr: float="1e-3", criterion: str="mse",
              early_stopping: bool=False, patience: int=50, device: str="cuda:0",
              log_per: int=1, fedprox_mu: float=0.0, reg1: float=0.0, reg2: float=0.0, max_grad_norm: float=0.0):

        best_model, best_loss, best_epoch = None, -1, -1
        train_loss_history, train_rmse_history = [], []
        val_loss_history, val_rmse_history, val_pinball_history = [], [], []
        if early_stopping:
            es_trace = True if log_per == 1 else False
            monitor = EarlyStopping(patience=patience, trace=es_trace)

        optimizer = self.get_optim(model=model, optim_name=optimizer, lr=lr)
        criterion = self.get_criterion(crit_name=criterion)

        global_weight_collector = copy.deepcopy(list(model.parameters()))

        for epoch in range(epochs):
            model.to(device)
            model.train()
            epochs_loss = []
            y_true_train, y_pred_train = [], []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                if fedprox_mu > 0.0:
                    fedprox_reg = 0.0
                    for param_index, param in enumerate(model.parameters()):
                        fedprox_reg += ( (fedprox_mu / 2) * T.norm((param - global_weight_collector[param_index])) ** 2 )
                    loss += fedprox_reg
                if reg1 > 0.0:
                    params = T.cat([p.view(-1) for name, p in model.named_parameters() if "bias" not in name])
                    loss += reg1 * T.norm(params, 1)
                if reg2 > 0.0:
                    params = T.cat([p.view(-1) for name, p in model.named_parameters() if "bias" not in name])
                    loss += reg2 * T.norm(params, 2)
                loss.backward()

                if max_grad_norm > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                epochs_loss.append(loss.detach())
                y_true_train.append(y.detach().cpu())
                y_pred_train.append(y_pred.detach().cpu())
                del loss, y_pred

            train_loss = T.stack(epochs_loss).mean().item()
            # Métricas de treino reaproveitam as predições já computadas no loop acima,
            # evitando um forward pass extra sobre train_loader. Por serem coletadas
            # durante o treino, refletem pesos em atualização ao longo da época (métrica
            # "em fluxo", como Keras/PyTorch Lightning reportam), não os pesos finais
            # da época como uma reavaliação a posteriori faria. val_loss/val_mse abaixo
            # não são afetados: continuam avaliados com os pesos finais da época.
            train_mse, train_rmse, train_mae, train_mape, train_r2, train_nrmse, mean_pinball = self.accumulate_metrics(
                T.cat(y_true_train, dim=0), T.cat(y_pred_train, dim=0)
            )
            val_loss, val_mse, val_rmse, val_mae, val_mape, val_r2, val_nrmse, mean_pinball, y_true_val, y_pred_val = self.test(model, val_loader, criterion, device)
            log(INFO, f"Participant: {self.cid} | Epoch {epoch + 1}/{epochs} | [Train]: loss {train_loss:.6f}, MSE: {train_mse:.6f} | [Val]: loss {val_loss:.6f}, MSE: {val_mse:.6f}")
            # wandb.log({
            #     "client/train_loss": train_loss,
            #     "client/val_loss": val_loss,
            #     "client/train_rmse": train_rmse,
            #     "client/val_rmse": val_rmse,
            #     "client/epoch": epoch + 1,
            #     "client/lr": lr
            # })
            self._log_prediction_slider(y_true_val, y_pred_val, epoch + 1)
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

        del optimizer
        del criterion
        gc.collect()
        return best_model, train_loss_history, val_loss_history


    def _log_prediction_slider(self, y_true, y_pred, epoch, max_points=96):
        y_t = y_true[:max_points, 0] if y_true.ndim > 1 else y_true[:max_points]
        y_p = y_pred[:max_points, 0] if y_pred.ndim > 1 else y_pred[:max_points]
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(y_t, label='Real.', color='#1f77b4', linewidth=2)
        ax.plot(y_p, label='Pred.', color='#ff7f0e', linewidth=2, linestyle='--')

        ax.set_title(f'{self.cid}')
        ax.set_ylabel('Grid')
        ax.set_xlabel('Obs.')
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=':', alpha=0.6)

        # wandb.log({
        #     "client/prediction_chart": wandb.Image(fig),
        #     "client/epoch": epoch
        # })
        plt.close(fig)


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
