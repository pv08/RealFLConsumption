import h5py
import torch as T
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from src.utils.functions import mkdir_if_not_exists

class ModelSerializer:
    def __init__(self, model, path: str, x_scaler=None, y_scaler=None):

        self.model = model
        self.path = path
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler


    def load(self):
        state_dict = OrderedDict()
        x_scaler, y_scaler = OrderedDict(), OrderedDict()
        with h5py.File(self.path, "r") as f:
            keys = list(f.keys())
            assert "model_weights" in keys
            # assert "x_scaler" in keys
            # assert "y_scaler" in keys
            model_weights = f["model_weights"]
            named_params = model_weights.keys()
            for k in named_params:
                state_dict[k] = model_weights[k][:]

            # x_scaler_state = f["x_scaler"]
            # named_params = x_scaler_state.keys()
            # for k in named_params:
            #     print(k)
            #     x_scaler[k] = x_scaler_state[k][:]
            #
            # y_scaler_state = f["y_scaler"]
            # named_params = y_scaler_state.keys()
            # for k in named_params:
            #     y_scaler[k] = y_scaler_state[k][:]

        state_dict = self.state_to_torch(state_dict)
        return state_dict

    def save(self, name):
        np_weights = self.state_to_numpy(self.model.state_dict())
        with h5py.File(f"{self.path}/{name}", "w") as f:
            group = f.create_group("model_weights", track_order=True)
            for k, v in np_weights.items():
                group[k] = v
            # group = f.create_group("x_scaler", track_order=True)
            # for k, v in self.x_scaler.items():
            #     group[k] = v
            # group = f.create_group("y_scaler", track_order=True)
            # for k, v in self.y_scaler.items():
            #     group[k] = v

    @staticmethod
    def state_to_numpy(model_state):
        assert type(model_state) in (dict, OrderedDict), \
            f"Model state must be of type dictionary. Received {type(model_state)}"
        k = next(iter(model_state))
        assert type(model_state[k]) in (T.tensor, T.Tensor, np.ndarray), \
            f"Model weights must be of type torch.tensor or numpy.ndarray. Received {type(model_state[k])}"
        if type(model_state[k]) == np.ndarray:
            return model_state
        np_ordered_dict = OrderedDict()
        for k, v in model_state.items():
            np_ordered_dict[k] = v.cpu().numpy().astype(np.float64)
        return np_ordered_dict

    @staticmethod
    def state_to_torch(model_state):
        assert type(model_state) in (dict, OrderedDict), \
            f"Model state must be of type dictionary. Received {type(model_state)}"
        k = next(iter(model_state))
        assert type(model_state[k]) in (T.tensor, T.Tensor, np.ndarray), \
            f"Model weights must be of type torch.tensor or numpy.ndarray. Received {type(model_state[k])}"
        if type(model_state[k]) in (T.tensor, T.Tensor):
            return model_state
        torch_ordered_dict = OrderedDict()
        for k, v in model_state.items():
            torch_ordered_dict[k] = T.tensor(v).float()
        return torch_ordered_dict