import torch as T
import pickle
import numpy as np
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional


class MongoDBDataset(Dataset):
    def __init__(self, _id, _type, mongo_uri, loc, buffer_size=1024):
        """
        Dataset com Buffer Inteligente.
        Carrega dados do Mongo em 'Chunks' (blocos) sob demanda.

        buffer_size: Quantas amostras carregar para a RAM por vez.
                     1024 é um bom equilíbrio entre RAM e Rede.
        """
        self.mongo_uri = mongo_uri
        self.db_name = "pecanstreet"
        self.collection_name = f"{loc}-samples"
        self.buffer_size = buffer_size

        # 1. Busca apenas os IDs (Muito leve, cabe na memória fácil)
        client = MongoClient(self.mongo_uri)
        col = client[self.db_name][self.collection_name]

        # Pega todos os IDs do cliente
        cursor = col.find({"client_id": str(_id), "type": _type}, {"_id": 1})
        self.all_ids = [doc["_id"] for doc in cursor]
        client.close()

        # Variáveis de controle do Buffer
        self.buffer = {}  # Dicionário para acesso rápido O(1)
        self.buffer_start_idx = -1
        self.buffer_end_idx = -1

    def __len__(self):
        return len(self.all_ids)

    def _load_buffer(self, target_idx):
        """
        Carrega um novo bloco de dados do Mongo centrado no índice alvo.
        """
        # Calcula qual 'página' precisamos carregar
        # Ex: se buffer=1000 e pedem o idx 2500, carregamos do 2000 ao 3000
        page_idx = target_idx // self.buffer_size
        start = page_idx * self.buffer_size
        end = min(start + self.buffer_size, len(self.all_ids))

        # Se já estamos com esse range na memória, não faz nada
        if start == self.buffer_start_idx:
            return

        # Pega a lista de IDs desse bloco
        ids_to_fetch = self.all_ids[start:end]

        # Abre conexão rápida
        client = MongoClient(self.mongo_uri)
        col = client[self.db_name][self.collection_name]

        # BUSCA EM LOTE (Bulk Fetch) - O Segredo da Performance
        # Usamos o operador $in para pegar 1000 documentos de uma vez
        cursor = col.find(
            {"_id": {"$in": ids_to_fetch}},
            {"X": 1, "y": 1, "_id": 1}
        )

        # Limpa buffer antigo e preenche o novo
        self.buffer.clear()
        for doc in cursor:
            # Deserializa e guarda no dict mapeado pelo _id
            self.buffer[doc["_id"]] = (pickle.loads(doc["X"]), pickle.loads(doc["y"]))

        client.close()

        # Atualiza ponteiros
        self.buffer_start_idx = start
        self.buffer_end_idx = end
        # print(f"[DEBUG] Buffer recarregado: índices {start} até {end}")

    def __getitem__(self, idx):
        # 1. Verifica se o índice está no buffer atual
        if not (self.buffer_start_idx <= idx < self.buffer_end_idx):
            self._load_buffer(idx)

        # 2. Recupera o _id real correspondente a este índice sequencial
        real_id = self.all_ids[idx]

        # 3. Pega do buffer
        # Nota: Mongo não garante ordem no $in, por isso usamos dict com _id
        try:
            X_raw, y_raw = self.buffer[real_id]
            return T.tensor(X_raw).float(), T.tensor(y_raw).float()
        except KeyError:
            # Caso raríssimo onde o ID existe na lista mas falhou no fetch
            # Retorna zeros ou gera erro (melhor tratar para não parar o treino)
            return T.zeros(1), T.zeros(1)


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
