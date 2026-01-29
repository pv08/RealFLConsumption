import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import numpy as np
import glob
from typing import Dict, List
from logging import INFO
from datetime import datetime
from src.utils.logger import log
from src.models.rnn import RNN
from src.models.lstm import LSTM
from src.models.gru import GRU
from src.models.cnn import CNN


def get_model(model: str, input_dim: int, out_dim: int, lags: int = 10, device:str = 'cpu'):
    if model == "rnn":
        model = RNN(device=device, input_dim=input_dim, rnn_hidden_size=128, num_rnn_layers=1, rnn_dropout=0.0,
                    layer_units=[128], num_outputs=out_dim, matrix_rep=True)
    elif model == "lstm":
        model = LSTM(device=device, input_dim=input_dim, lstm_hidden_size=128, num_lstm_layers=1,
                     lstm_dropout=0.0,
                     layer_units=[128], num_outputs=out_dim, matrix_rep=True)
    elif model == "gru":
        model = GRU(device=device, input_dim=input_dim, gru_hidden_size=128, num_gru_layers=1, gru_dropout=0.0,
                    layer_units=[128], num_outputs=out_dim, matrix_rep=True)
    elif model == "cnn":
        model = CNN(device=device, num_features=input_dim, lags=lags, out_dim=out_dim)
    else:
        raise NotImplementedError(
            "Specified model is not implemented. Please define your own model or choose one from ['mlp', 'rnn', 'lstm', 'gru', 'cnn', 'da_encoder_decoder']")
    return model


def mkdir_if_not_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def make_default_dirs(model_name: str):
    mkdir_if_not_exists(f'etc/fl/ckpts/{model_name}/server/best/')
    mkdir_if_not_exists(f'etc/fl/ckpts/{model_name}/local/best/')
    mkdir_if_not_exists(f'etc/fl/ckpts/{model_name}/local/epochs/')

    mkdir_if_not_exists(f'etc/fl/logs/{model_name}/server/')
    mkdir_if_not_exists(f'etc/fl/logs/{model_name}/local')

    mkdir_if_not_exists(f'etc/fl/results/{model_name}/preds')

    mkdir_if_not_exists(f'etc/in/ckpts/{model_name}/best')
    mkdir_if_not_exists(f'etc/in/ckpts/{model_name}/epochs')

    mkdir_if_not_exists(f'etc/in/logs/{model_name}')
    mkdir_if_not_exists(f'etc/in/results/{model_name}')


def save_json_file(save_path: str, values):
    if type(values) == list:
        with open(save_path, 'wb') as f:
            np.save(f, np.array(values))
    elif type(values) == dict:
        with open(save_path, "w") as outfile:
            json.dump(values, outfile)
    else:
        raise TypeError(f"Type {type(values)} not recognized: {values}")
    log(INFO, f"json file created on {save_path}")

def convert_time_to_float(var):
    try:
        obj = datetime.strptime(var, "%I:%M %p")
        return obj.hour + obj.minute / 60.0
    except:
        return 0.0

def get_params(alg):
    if alg == "fedprox":
        return {"mu": 0.01}
    elif alg == "fednova":
        return {"rho": 0.}
    elif alg == "fedadagrad":
        return {"beta_1": 0., "eta": 0.1, "tau": 1e-2}
    elif alg == "fedyogi":
        return {"beta_1": 0.9, "beta_2": 0.99, "eta": 0.01, "tau": 1e-3}
    elif alg == "fedadam":
        return {"beta_1": 0.9, "beta_2": 0.99, "eta": 0.01, "tau": 1e-3}
    elif alg == "fedavgm":
        return {"server_momentum": 0., "server_lr": 1.}
    else:
        return None

def plot_train_curve(train_history, test_history, title, fig_name, model_name):
    plt.title(title)
    plt.plot(train_history, label='Train')
    plt.plot(test_history, label='Validation')
    plt.legend()
    
    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/results/')
    mkdir_if_not_exists('etc/results/individual')
    mkdir_if_not_exists(f'etc/results/individual/{model_name}')
    plt.savefig(f'etc/results/individual/{model_name}/{fig_name}.png')
    plt.close()

def plot_global_losses(values: List[float], model_name):
    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/results/')
    mkdir_if_not_exists('etc/results/imgs')
    mkdir_if_not_exists(f'etc/results/imgs/{model_name}')

    plt.title(f'Loss per round')
    plt.plot(values, marker='^')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    
    plt.savefig(f'etc/results/imgs/{model_name}/global_loss.png')
    plt.close()


def plot_global_metrics(history: Dict[str, List[np.float64]], model_name):
    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/results/')
    mkdir_if_not_exists('etc/results/imgs')
    mkdir_if_not_exists(f'etc/results/imgs/{model_name}')

    for metric, value in history.items():
        plt.title(f'{metric} evaluation per round')
        plt.plot(value, marker='^')
        plt.xlabel('Round')
        plt.ylabel('Error')
        
        plt.savefig(f'etc/results/imgs/{model_name}/{metric}.png')
        plt.close()

def plot_local_train_rounds(history, model_name):
    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/results/')
    mkdir_if_not_exists('etc/results/imgs')
    mkdir_if_not_exists(f'etc/results/imgs/{model_name}')

    cids = [participant for participant in history.keys()]
    counts = [len(participant) for participant in history.values()]
    plt.title('Count of local trainings')
    plt.bar(cids, counts)
    plt.xlabel('Participants')
    plt.ylabel('Training times')
    plt.xticks(rotation=45)
    
    plt.savefig(f'etc/results/imgs/{model_name}/training_times.png')
    plt.close()

def plot_test_prediction(y_true, y_pred, cid):
    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/results/')
    mkdir_if_not_exists('etc/results/imgs/')
    mkdir_if_not_exists('etc/results/imgs/preds/')
    plt.title(f"Prediction of {cid}")
    plt.plot(y_true, label='True')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.savefig(f'etc/results/imgs/preds/{cid}_pred.png')
    plt.close()


def transform_preds_test(y_pred_test):
    if not isinstance(y_pred_test, np.ndarray):
        y_pred_test = y_pred_test.cpu().numpy()
    return y_pred_test

def round_predictions_test(y_pred_test, dims):
    # round to closest integer
    if dims is None or len(dims) == 0:
        return y_pred_test
    for dim in dims:
        y_pred_test[:, dim] = np.rint(y_pred_test[:, dim])
    return y_pred_test



def inverse_transform_test(
        y_test, y_pred_test,
        y_scaler=None,
        round_preds=False,
        dims=None):
    y_pred_test = transform_preds_test(y_pred_test)

    if y_scaler is not None:
        y_test = y_scaler.inverse_transform(y_test)
        y_pred_test = y_scaler.inverse_transform(y_pred_test)

    # to zeroes
    # y_pred_test[y_pred_test < 0.] = 0.

    if round_preds:
        y_pred_test = round_predictions_test(y_pred_test, dims)

    return y_test, y_pred_test


def make_plot(y_true, y_pred, model_name,
              title,
              feature_names=None,
              client=None, individual=False):
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(y_pred.shape[1])]
    assert len(feature_names) == y_pred.shape[1]

    for i in range(y_pred.shape[1]):
        client_preds = {"y_true": y_true[:, i].tolist(), "y_pred": y_pred[:, i].tolist(), "client": client}
        mkdir_if_not_exists('etc/')
        mkdir_if_not_exists('etc/results/')
        mkdir_if_not_exists(f'etc/results/{model_name}/')
        mkdir_if_not_exists(f'etc/results/{model_name}/predictions/')
        mkdir_if_not_exists(f'etc/results/{model_name}/predictions/individual/')
        path = f'etc/results/{model_name}/predictions/{client}.json'
        if individual:
            path = f'etc/results/{model_name}/predictions/individual/{client}.json'
        with open(path, 'w') as file:
            json.dump(client_preds, file)
        log(INFO, f"json file created on etc/results/{model_name}/predictions/{client}.json")

        plt.figure(figsize=(8, 6))
        plt.ticklabel_format(style='plain')
        plt.plot(y_true[:, i][:96], label="Actual")
        plt.plot(y_pred[:, i][:96], label="Predicted")
        if client is not None:
            plt.title(f"[{client} {title}] {feature_names[i]} prediction")
        else:
            plt.title(f"[{title}] {feature_names[i]} prediction")
        plt.legend()

        mkdir_if_not_exists('etc/')
        mkdir_if_not_exists('etc/imgs')
        mkdir_if_not_exists('etc/imgs/preds/')
        mkdir_if_not_exists(f'etc/imgs/preds/{model_name}')
        mkdir_if_not_exists(f'etc/imgs/preds/{model_name}/individual/')
        if not individual:
            plt.savefig(f'etc/imgs/preds/{model_name}/{client}.png')
        else:
            plt.savefig(f'etc/imgs/preds/{model_name}/individual/{client}.png')
        plt.close()

def generate_train_test_collections(path):
    files = glob.glob(f"{path}/*.csv", recursive=True)
    mkdir_if_not_exists(f'{path}/train/')
    mkdir_if_not_exists(f'{path}/test/')
    for file in files:
        filename = file.split('/')[-1]
        data = pd.read_csv(file)
        data = data.sort_values(by='Date').reset_index(drop=True)
        data['Date'] = pd.to_datetime(data['Date'])
        train_df = data.loc[(data['Date'] >= '2019-01-01') & (data['Date'] < '2019-12-30')].reset_index(drop=True)
        train_df.to_csv(f'dataset/FLHousesData/train/{filename}', index_label=False)
        test_df = data.loc[data['Date'] >= '2019-12-30'].reset_index(drop=True)
        test_df.to_csv(f'dataset/FLHousesData/test/{filename}', index_label=False)
    log(INFO, f"Train and test collections saved on {path}")

