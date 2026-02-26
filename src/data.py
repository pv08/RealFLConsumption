import torch as T
import pickle
import numpy as np
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional

class LocalFileDataset(Dataset):
    def __init__(self, client_id, _type, data_path="dataset/pecanstreet/15min/austin/train/"):
        # Caminhos para os arquivos pré-processados
        self.X_path = f"{data_path}/{client_id}-{_type}-X.npy"
        self.y_path = f"{data_path}/{client_id}-{_type}-y.npy"

        # Mapeia o arquivo em memória (não carrega na RAM ainda)
        self.X = T.from_numpy(np.load(self.X_path)).float()
        self.y = T.from_numpy(np.load(self.y_path)).float()


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # O dado só é lido do disco para a RAM no momento do acesso
        return self.X[idx], self.y[idx]