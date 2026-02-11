import torch as T
import pickle
import numpy as np
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional

from paths import TRAIN_DIR


class MongoDBDataset(Dataset):
    def __init__(self, _id, _type, mongo_uri, loc, buffer_size=1024):
        self.mongo_uri = mongo_uri
        self.db_name = "pecanstreet"
        self.collection_name = f"{loc}-samples"
        self.buffer_size = buffer_size

        client = MongoClient(self.mongo_uri)
        col = client[self.db_name][self.collection_name]

        # Pega todos os IDs do cliente
        cursor = col.find(
            {"client_id": str(_id), "type": _type},
            {"X": 1, "y": 1, "_id": 0}
        )
        X_list = []
        y_list = []
        for doc in cursor:
            X_list.append(pickle.loads(doc["X"]))
            y_list.append(pickle.loads(doc["y"]))
        client.close()

        self.X = T.tensor(np.array(X_list)).float()
        self.y = T.tensor(np.array(y_list)).float()

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LocalFileDataset(Dataset):
    def __init__(self, client_id, _type, data_path=TRAIN_DIR):
        self.X_path = data_path / f"{client_id}-{_type}-X.npy"
        self.y_path = data_path / f"{client_id}-{_type}-y.npy"

        self.X = T.from_numpy(np.load(self.X_path)).float()
        self.y = T.from_numpy(np.load(self.y_path)).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]