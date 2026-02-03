import torch as T
import pickle
import numpy as np
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional


class MongoDBDataset(Dataset):
    def __init__(self, _id, _type, mongo_uri, loc, buffer_size=1024):
        self.mongo_uri = mongo_uri
        self.db_name = "pecanstreet"
        self.collection_name = f"{loc}-samples"
        self.buffer_size = buffer_size

        client = MongoClient(self.mongo_uri)
        col = client[self.db_name][self.collection_name]

        # Pega todos os IDs do cliente
        cursor = col.find({"client_id": str(_id), "type": _type}, {"_id": 1})
        self.all_ids = [doc["_id"] for doc in cursor]
        X_list = [pickle.loads(doc["X"]) for doc in cursor]
        y_list = [pickle.loads(doc["y"]) for doc in cursor]
        client.close()

        self.X = T.tensor(np.array(X_list)).float()
        self.y = T.tensor(np.array(y_list)).float()
        
    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TimeSeriesLoader:
    def __init__(self, X, y,  num_lags, num_features, indices, batch_size, shuffle, num_workers):
        self.X = T.tensor(X).float()
        self.y = T.tensor(y).float()
        self.num_lags = num_lags
        self.num_features = num_features

        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers



    def get_dataloader(self):
        tensor_dataset = TimeSeriesDataset(self.X, self.y, self.num_lags, self.num_features,
                                           self.indices)
        return DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=False)





class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, num_lags: int=10, num_features: int=11,
                 indices: List[int]=[0]):
        assert X.size(0) == y.size(0), "Size mismatch between tensors"

        self.X = X
        self.y = y
        self.num_lags = num_lags
        self.num_features = num_features
        self.indices = indices

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        # if idx == 0:
        #     tmp_X = self.X[idx]
        #     if len(self.X.shape) < 3:
        #         tmp_X = tmp_X.view(self.num_lags, self.num_features, 1)
        #     y_hist = []
        #     for i, lag in enumerate(tmp_X):
        #         if i == 0:
        #             pad = T.zeros_like(lag[self.indices])
        #             y_hist.append(pad.reshape(1, -1))
        #         else:
        #             y_hist.append((tmp_X[i - 1][self.indices].reshape(1, -1)))
        #     y_hist = T.cat(y_hist)
        # elif idx < self.num_lags + 1:
        #     last_obs = self.X[idx - 1]
        #     if len(self.X.shape) < 3:
        #         last_obs = last_obs.view(self.num_lags, self.num_features, 1)
        #     y_hist = []
        #     for i, lag in enumerate(last_obs):
        #         y_hist.append(last_obs[i][self.indices].reshape(1, -1))
        #     y_hist = T.cat(y_hist)
        # else:
        #     y_hist = self.y[idx - self.num_lags - 1: idx - 1]
        #
        return self.X[idx], self.y[idx]
