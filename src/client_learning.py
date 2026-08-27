import os
import pickle
import random
import copy
import hashlib
import json
import sys
import time
import numpy as np
import torch as T
import torch.nn as nn
import math
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Union, Any, Dict
from logging import INFO, DEBUG
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_pinball_loss
from collections import OrderedDict, defaultdict
from src.models.timeVAE.timevae import TimeVAE
from src.utils.functions import inverse_transform_test, mkdir_if_not_exists, seed_all
from src.utils.logger import log
from src.data import LocalFileDataset
from src.utils.early_stopping import EarlyStopping

def _fmt_dur(seconds: float) -> str:
    """Duração legível para as linhas de andamento: 45s, 3m12s, 1h04m."""
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


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

        if hparams:
            for k, v in hparams.items():
                setattr(self.args, k, v)

        self.model = None

    def get_latent_space(self, latent_dim: int=8, epochs: int=8, mode: str="fixed", samples_per_week: int=672):
        """Extrai a assinatura latente do cliente a partir do encoder do TimeVAE.

        Treina (ou carrega do cache) um TimeVAE por cliente e produz o `z_mean` de cada
        janela de treino. Modos:
          - "fixed":  média global dos `z_mean` -> um único vetor flat (para clustering fixo).
          - "weekly": fatia as janelas em blocos consecutivos de `samples_per_week` amostras e
                      devolve `{f"week_{i}": média_do_bloco}` (i a partir de 1), para o clustering
                      semanal. Requer ordem cronológica das janelas (o `.npy` de treino já é
                      temporalmente ordenado).
        O checkpoint depende apenas de `latent_dim`, então os dois modos compartilham o mesmo ckpt.
        """
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

        # Loader ORDENADO (shuffle=False): no modo semanal a ordem cronológica das janelas define
        # a que semana cada `z_mean` pertence; no modo fixo a ordem é irrelevante (só a média global).
        ordered_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=False,
                                    num_workers=self.args.num_workers, pin_memory=True)
        with T.no_grad():
            for X, _ in ordered_loader:
                X = X.to(self.args.device)
                X = X.reshape(X.size(0), X.size(1), X.size(2))

                z_mean, _, _ = timevae.encoder(X)
                latent_vectors.append(z_mean.cpu())

        all_latents = T.cat(latent_vectors, dim=0)

        if mode == "weekly":
            num_samples = all_latents.size(0)
            num_weeks = max(1, math.ceil(num_samples / samples_per_week))
            weekly_signature = {}
            for i in range(num_weeks):
                start = i * samples_per_week
                end = min(start + samples_per_week, num_samples)
                weekly_signature[f"week_{i + 1}"] = all_latents[start:end].mean(dim=0).numpy()
            log(INFO, f"{self.cid} produced {num_weeks} weekly latent signatures "
                      f"({num_samples} windows / {samples_per_week} per week)")
            return weekly_signature

        client_signature = all_latents.mean(dim=0).numpy()
        return client_signature

    # ------------------------------------------------------------------
    # Generative TimeVAE (TSTR/TRTS evaluation) — sintetiza janelas
    # completas (X + y) modelando a série multivariada de comprimento
    # num_lags+output_dim, sem o artifício concat_X_y. Ver eval_timevae.py.
    # ------------------------------------------------------------------
    def _derive_consumption_index(self):
        """Descobre o canal do consumo em X pelo casamento exato X[i,-1,c] == y[i-1].

        Robusto e independente da ordem das colunas: y[t]=consumption[t] é o próximo
        valor do canal de consumo, então o passo mais recente da janela i (X[i,-1])
        no canal de consumo é igual a y[i-1]. Cacheia o índice."""
        if getattr(self, "_cons_idx", None) is not None:
            return self._cons_idx
        X = self.train_dataset.X.numpy()
        y = self.train_dataset.y.numpy()
        X = X[..., 0] if X.ndim == 4 else X  # (N, lags, feat)
        yv = y[:-1, 0]
        for c in range(X.shape[2]):
            if np.allclose(X[1:, -1, c], yv, atol=1e-6):
                self._cons_idx = c
                log(INFO, f"{self.cid}: consumption channel derived at index {c}")
                return c
        raise ValueError(
            f"Could not derive consumption channel for client {self.cid}: "
            f"no channel c satisfies X[i,-1,c] == y[i-1] exactly."
        )

    def channel_names(self) -> List[str]:
        """Nomes dos canais na ORDEM do índice de canal da janela.

        O `x_scaler` foi ajustado sobre o frame de features original, mas
        `Processing.generate_time_lags` inverte as colunas (`df[df.columns[::-1]]`) antes de
        `to_timeseries_rep` remontar `(N, lags, feat)`. O efeito líquido é que o canal j da
        janela corresponde a `feature_names_in_[F-1-j]`. Verificado empiricamente: `minute`
        (4 valores distintos) cai em j=35, `hour` (24) em j=36, `second`/`year` constantes,
        e `consumption` em j=40 — o mesmo índice que `_derive_consumption_index` deriva por
        casamento exato com o alvo.
        """
        if getattr(self, "_channel_names", None) is not None:
            return self._channel_names
        names = getattr(self.x_scaler, "feature_names_in_", None)
        if names is None:
            raise ValueError(
                f"x_scaler of client {self.cid} has no feature_names_in_; cannot name channels. "
                f"Re-run the preprocessing (migrate_data_numpy.py) so the scaler is fit on a DataFrame."
            )
        self._channel_names = list(names)[::-1]
        # Trava de segurança: se o pré-processamento mudar a ordem das colunas, isso falha alto
        # em vez de silenciosamente pesar/gerar o canal errado.
        derived, by_name = self._derive_consumption_index(), self._channel_names.index("consumption")
        if derived != by_name:
            raise ValueError(
                f"Channel order mismatch for client {self.cid}: 'consumption' is at index {by_name} "
                f"by name but at {derived} by exact target matching. The preprocessing column order "
                f"changed — fix channel_names() before using channel weights or channel subsets."
            )
        return self._channel_names

    def channel_index(self, name: str) -> int:
        """Índice de canal da janela para um nome de feature (ver channel_names)."""
        names = self.channel_names()
        if name not in names:
            raise ValueError(f"Unknown channel {name!r} for client {self.cid}. Available: {names}")
        return names.index(name)

    def build_channel_wt(self, weights: Optional[Dict[str, float]],
                         channels: Optional[List[int]]=None) -> Optional[T.Tensor]:
        """Monta o vetor de pesos por canal a partir de {nome: peso}; canais não citados ficam
        em 1.0. Devolve None quando todos os pesos são 1.0, para que o caminho uniforme continue
        sendo exatamente o de antes (mesma loss e mesmo hash de checkpoint).

        `channels` restringe o vetor a um subconjunto de índices (geração condicional, em que a
        loss só cobre os canais modelados). Peso posto num canal de fora do subconjunto some,
        que é o correto: ele não entra na loss."""
        if not weights:
            return None
        w = T.ones(self.input_dim, dtype=T.float32)
        for name, value in weights.items():
            w[self.channel_index(name)] = float(value)
        if channels is not None:
            w = w[T.as_tensor(list(channels), dtype=T.long)]
        if T.all(w == 1.0):
            return None
        return w

    def generative_channel_split(self, gen_channels=None):
        """Partição dos canais para a geração condicional: (gen_idx, cond_idx).

        `gen_idx` são os canais que o gerador MODELA (endógenos: consumo, geração, tensões e
        derivados); `cond_idx` é o complemento (clima e calendário), que condiciona o decoder e
        é copiado de uma janela real na amostragem — são exógenos ou determinísticos, não faz
        sentido inventá-los. Retorna (None, None) quando a geração condicional está desligada."""
        gen_channels = gen_channels if gen_channels is not None else getattr(self.args, "gen_channels", None)
        if not gen_channels:
            return None, None
        gen_idx = sorted({self.channel_index(name) for name in gen_channels})
        cond_idx = [j for j in range(self.input_dim) if j not in set(gen_idx)]
        if not cond_idx:
            raise ValueError(
                f"Client {self.cid}: --gen_channels covers all {self.input_dim} channels, so there is "
                f"nothing left to condition on. Drop the flag to train an unconditional generator."
            )
        cons = self._derive_consumption_index()
        if cons not in gen_idx:
            raise ValueError(
                f"Client {self.cid}: --gen_channels must include 'consumption' (channel {cons}) — it is "
                f"the forecasting target, so y_syn would otherwise be copied from real data and TSTR "
                f"would measure nothing. Got channels {sorted(gen_idx)}."
            )
        return gen_idx, cond_idx

    def _stitch_windows(self, X):
        """Monta janelas generativas de comprimento num_lags+output_dim a partir de
        janelas reais consecutivas (stride-1): W[i] = concat(X[i], X[i+1..i+H][-1]).
        O passo extra usa features REAIS da(s) janela(s) seguinte(s) — nada fabricado.
        X: (N, lags, feat). Retorna (N-H, lags+H, feat)."""
        H = self.output_dim
        N = X.shape[0]
        if N <= H:
            return np.empty((0, self.args.num_lags + H, X.shape[2]), dtype=X.dtype)
        base = X[:N - H]  # (N-H, lags, feat)
        extra = np.stack([X[1 + k: N - H + 1 + k, -1, :] for k in range(H)], axis=1)  # (N-H, H, feat)
        return np.concatenate([base, extra], axis=1)

    def _midnight_indices(self, W):
        """Índices das janelas cujo passo 0 cai em 00:00, lidos dos canais `hour`/`minute`.

        Os canais estão escalados, mas o min-max é monótono, então o menor valor do canal
        corresponde a hour=0 / minute=0. A seleção é por PREDICADO, não por passo aritmético:
        a série do cliente tem descontinuidades (no 661 há um salto de 92 passos em vez de 96),
        então `W[start::96]` sairia de fase depois do buraco — só 114 de 325 janelas começavam
        de fato à meia-noite."""
        h, m = self.channel_index("hour"), self.channel_index("minute")
        hour0, min0 = W[:, 0, h], W[:, 0, m]
        return np.flatnonzero(np.isclose(hour0, hour0.min()) & np.isclose(min0, min0.min()))

    def _subsample_windows(self, W, stride: int, align_to_day: bool):
        """Reamostra as janelas stride-1 produzidas por `_stitch_windows`.

        A `SeasonalLayer` do TimeVAE indexa por POSIÇÃO NA JANELA. Com stride 1 cada janela
        começa numa fase arbitrária do dia, então a base sazonal não corresponde a hora do dia;
        com `align_to_day=True` toda janela começa em 00:00 e a sazonalidade passa a ser
        interpretável. Custo: ~96x menos janelas. `stride` é aplicado depois — sobre as janelas
        de meia-noite (isto é, em dias) quando `align_to_day`, senão em passos de 15 min."""
        if len(W) == 0 or (stride == 1 and not align_to_day):
            return W
        if not align_to_day:
            return W[::stride]
        idx = self._midnight_indices(W)
        if len(idx) == 0:
            log(INFO, f"{self.cid}: no window starts at 00:00; day alignment skipped for this split")
            return W[::stride]
        return W[idx[::stride]]

    def _apply_gen_holdout(self, W, split: str, holdout: str):
        """Reserva um bloco CONTÍGUO no fim da série para avaliar o gerador fora da amostra.

        O corte 80/20 de `_generative_loaders` é aleatório sobre janelas de stride 1, então uma
        janela "reservada" quase sempre tem vizinhas no treino compartilhando `seq_len-1` passos
        — o modelo praticamente já a viu, e o gap medido subestima a memorização. Um bloco
        contíguo no fim resolve isso, desde que se descarte também um COLCHÃO de `seq_len-1`
        janelas antes dele: sem o colchão, as últimas janelas de treino ainda invadem o bloco.

        `holdout`: 'exclude' (treino, sem o bloco nem o colchão) | 'only' (só o bloco) |
        'all' (série inteira, comportamento de antes). Só se aplica a split='train'."""
        days = getattr(self.args, "gen_holdout_days", 0) or 0
        if split != "train" or days <= 0 or holdout == "all" or len(W) == 0:
            return W

        seq_len = self.args.num_lags + self.output_dim
        stride = getattr(self.args, "window_stride", 1) or 1
        aligned = getattr(self.args, "align_to_day", False)
        # Passos de 15 min entre janelas consecutivas de W, já contando a reamostragem.
        step = (96 * stride) if aligned else stride
        n_hold = max(1, round(days * 96 / step))
        buffer = int(np.ceil((seq_len - 1) / step))

        if holdout == "only":
            return W[-n_hold:]
        n_train = len(W) - n_hold - buffer
        if n_train < 2:
            raise ValueError(
                f"Client {self.cid}: --gen_holdout_days {days} reserves {n_hold} window(s) plus a "
                f"{buffer}-window buffer out of {len(W)}, leaving {n_train} for training. Reduce "
                f"the holdout."
            )
        return W[:n_train]

    def build_generative_windows(self, split: str="train", window_stride: int=None,
                                 align_to_day: bool=None, holdout: str="exclude"):
        """Retorna (W, cons_idx). split='train' usa train+val (costurados separadamente
        para não criar janela espúria na fronteira); split='test' usa o conjunto de teste.

        `window_stride`/`align_to_day` (default: os de `args`, ou 1/False) controlam a
        reamostragem — aplicada por parte, para não misturar as fases de train e val.
        `holdout` controla o bloco contíguo reservado (ver `_apply_gen_holdout`)."""
        window_stride = window_stride if window_stride is not None else getattr(self.args, "window_stride", 1)
        align_to_day = align_to_day if align_to_day is not None else getattr(self.args, "align_to_day", False)
        cons_idx = self._derive_consumption_index()
        if split == "train":
            parts = [
                self._stitch_windows(self.train_dataset.X.numpy()[..., 0]),
                self._stitch_windows(self.val_dataset.X.numpy()[..., 0]),
            ]
        elif split == "test":
            test_ds = LocalFileDataset(client_id=self.args.filter_bs, _type="test", data_path=self.args.test_path)
            parts = [self._stitch_windows(test_ds.X.numpy()[..., 0])]
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        parts = [self._subsample_windows(p, window_stride, align_to_day) for p in parts]
        kept = [p for p in parts if len(p)]
        W = np.concatenate(kept, axis=0) if kept else parts[0]
        return self._apply_gen_holdout(W, split, holdout), cons_idx

    def _build_generative_vae(self, latent_dim: int, hidden_dims=None, gen_channels=None,
                              cond_dim: int=None):
        from src.models.timeVAE.timevae import TimeVAE
        seq_len = self.args.num_lags + self.output_dim
        gen_idx, cond_idx = self.generative_channel_split(gen_channels)
        cond_dim = cond_dim if cond_dim is not None else getattr(self.args, "cond_dim", 32)
        return TimeVAE(hidden_sizes=hidden_dims or self.args.hidden_dims, trend_poly=self.args.trend_poly,
                       custom_seats=self.args.custom_seats, use_residual_conn=self.args.use_residual_conn,
                       seq_len=seq_len, feat_dim=self.input_dim, latent_dim=latent_dim, device=self.args.device,
                       gen_idx=gen_idx, cond_idx=cond_idx,
                       cond_dim=cond_dim if gen_idx is not None else 0)

    def _generative_loaders(self, split: str, batch_size: int=None, val_shuffle: bool=None):
        """Loaders (W, W) para treinar o VAE gerador. train_timevae itera 'for X, y' e só usa X.

        `val_shuffle=True` (default) embaralha antes de separar a validação, como a referência
        (`split_data(shuffle=True)`). Com o corte cronológico anterior o gerador nunca via o
        período mais recente e a val loss vinha de outro regime, o que enviesava tanto a escolha
        do best_model quanto a HPO. A permutação usa `args.seed`, então é reprodutível."""
        bs = batch_size or self.args.batch_size
        val_shuffle = val_shuffle if val_shuffle is not None else getattr(self.args, "gen_val_shuffle", True)
        W, _ = self.build_generative_windows(split)
        n_val = max(1, int(0.2 * len(W)))
        if len(W) - n_val < 1:
            stride = getattr(self.args, "window_stride", 1)
            aligned = getattr(self.args, "align_to_day", False)
            raise ValueError(
                f"Client {self.cid}: split '{split}' yielded {len(W)} generative window(s) — not enough "
                f"to train a generator (needs at least 2: one for training, one for validation). "
                f"Current window options: align_to_day={aligned}, window_stride={stride}. "
                f"The test split spans roughly one day, so day alignment collapses it to a single "
                f"window; use --mode TSTR (which fits the generator on the train split) or drop "
                f"--align_to_day for this split."
            )
        W = T.from_numpy(W).float()
        if val_shuffle:
            perm = T.randperm(len(W), generator=T.Generator().manual_seed(self.args.seed))
            W = W[perm]
        W_tr, W_val = W[:-n_val], W[-n_val:]
        tr = DataLoader(TensorDataset(W_tr, W_tr), batch_size=bs, shuffle=True, num_workers=self.args.num_workers)
        val = DataLoader(TensorDataset(W_val, W_val), batch_size=bs, shuffle=False, num_workers=self.args.num_workers)
        return tr, val

    def _gen_config_hash(self, cfg: dict) -> str:
        """Hash curto de tudo que muda os PESOS resultantes do gerador.

        O nome do checkpoint codificava só `latent_dim`, então um gerador treinado com outra loss
        ou outra arquitetura era recarregado do cache silenciosamente (ou estourava no
        `load_state_dict`). Com o hash no nome, cada configuração tem o seu arquivo."""
        blob = json.dumps(cfg, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:8]

    def train_generative_timevae(self, latent_dim: int=8, epochs: int=8, split: str="train",
                                 lr: float=1e-3, beta: float=1.0, kl_anneal_frac: float=0.0,
                                 reconstruction_wt: float=None, hidden_dims=None,
                                 mean_axis_wt: float=1.0, var_wt: float=0.0, grad_wt: float=0.0,
                                 mean_time_wt: float=None, mean_feat_wt: float=None,
                                 channel_wt: Optional[T.Tensor]=None, batch_size: int=None,
                                 log_every: Optional[int]=None,
                                 cache: bool=True, force: bool=False, tag: str=""):
        """Treina (ou carrega do cache) um TimeVAE gerador (seq_len=num_lags+output_dim) sobre as
        janelas costuradas do split. Hiperparâmetros generativos (lr/beta/kl_anneal_frac/
        reconstruction_wt/hidden_dims/batch_size) e forma da loss (mean_axis_wt/mean_time_wt/
        mean_feat_wt/var_wt/grad_wt/channel_wt) configuráveis. `tag` distingue famílias de
        checkpoint (ex.: '-opt'); o hash de configuração distingue o resto. `cache=False` treina
        sem ler/gravar; `force=True` retreina."""
        gen = self._build_generative_vae(latent_dim, hidden_dims)
        mkdir_if_not_exists(f'etc/TimeVAE/{self.args.loc}/ckpt/')
        mkdir_if_not_exists(f'etc/TimeVAE/{self.args.loc}/logs/')

        gen_idx, _ = self.generative_channel_split()
        cfg = {"latent_dim": latent_dim, "epochs": epochs, "split": split, "lr": lr, "beta": beta,
               "kl_anneal_frac": kl_anneal_frac, "reconstruction_wt": reconstruction_wt,
               "hidden_dims": list(hidden_dims or self.args.hidden_dims), "trend_poly": self.args.trend_poly,
               "custom_seas": self.args.custom_seats, "use_residual_conn": self.args.use_residual_conn,
               "seq_len": self.args.num_lags + self.output_dim, "feat_dim": self.input_dim,
               "mean_axis_wt": mean_axis_wt, "mean_time_wt": mean_time_wt, "mean_feat_wt": mean_feat_wt,
               "var_wt": var_wt, "grad_wt": grad_wt,
               "channel_wt": None if channel_wt is None else [round(float(v), 6) for v in channel_wt],
               "batch_size": batch_size or self.args.batch_size,
               "window_stride": getattr(self.args, "window_stride", 1),
               "align_to_day": getattr(self.args, "align_to_day", False),
               "gen_val_shuffle": getattr(self.args, "gen_val_shuffle", True),
               # A seed entra no hash porque muda os pesos resultantes (init, ordem dos batches,
               # split de validação). Sem ela, rodar o mesmo braço com sementes diferentes
               # recarregaria o checkpoint da primeira e as réplicas sairiam idênticas.
               "seed": self.args.seed}
        # As chaves da geração condicional só entram QUANDO ela está ligada: acrescentá-las
        # sempre (ainda que como None) mudaria o hash de toda configuração não-condicional e
        # invalidaria os checkpoints dos braços A0-A5 já treinados.
        if gen_idx is not None:
            cfg["gen_channels"] = gen_idx
            cfg["cond_dim"] = getattr(self.args, "cond_dim", 32)
        # Mesma regra: a chave só entra quando o bloco reservado existe, para não invalidar os
        # checkpoints dos braços treinados sem ele.
        if getattr(self.args, "gen_holdout_days", 0):
            cfg["gen_holdout_days"] = self.args.gen_holdout_days
        cond_tag = "-cond" if gen_idx is not None else ""
        stem = f'{self.cid}-gen-{split}{tag}{cond_tag}-ld{latent_dim}-{self._gen_config_hash(cfg)}'
        ckpt = f'etc/TimeVAE/{self.args.loc}/ckpt/{stem}.pth'

        if cache and not force and os.path.exists(ckpt):
            log(INFO, f"{self.cid}'s generative TimeVAE ({split}{tag}) found at {ckpt}. Loading state dict")
            gen.load_state_dict(T.load(ckpt, map_location=self.args.device))
            return gen

        # Fallback OPT-IN para o nome antigo (só latent_dim), para reaproveitar geradores treinados
        # antes do hash existir. É opt-in de propósito: se fosse automático, um checkpoint legado
        # sombrearia TODA configuração nova (channel_wt, trend_poly, ...) — o run carregaria o
        # modelo velho em vez de treinar o pedido, sem erro nenhum. Só use para reproduzir a
        # baseline histórica, nunca para comparar formas de loss.
        legacy = f'etc/TimeVAE/{self.args.loc}/ckpt/{self.cid}-gen-{split}{tag}-latent_dim_{latent_dim}.pth'
        if cache and not force and getattr(self.args, "allow_legacy_ckpt", False) and os.path.exists(legacy):
            log(INFO, f"{self.cid}: no checkpoint for the current config ({ckpt}); --allow_legacy_ckpt is "
                      f"set, so loading the LEGACY checkpoint {legacy}. Its training config is NOT "
                      f"verifiable and probably does not match the flags you just passed.")
            gen.load_state_dict(T.load(legacy, map_location=self.args.device))
            return gen

        log(INFO, f"{self.cid}'s generative TimeVAE ({split}{tag}, latent_dim {latent_dim}) training {cfg}")
        tr_loader, val_loader = self._generative_loaders(split, batch_size=batch_size)
        kl_anneal_epochs = int(kl_anneal_frac * epochs)
        gen, train_val_log = self.train_timevae(gen, tr_loader, val_loader, epochs, lr=lr, beta=beta,
                                                kl_anneal_epochs=kl_anneal_epochs, reconstruction_wt=reconstruction_wt,
                                                mean_axis_wt=mean_axis_wt, var_wt=var_wt, grad_wt=grad_wt,
                                                mean_time_wt=mean_time_wt, mean_feat_wt=mean_feat_wt,
                                                channel_wt=channel_wt, log_every=log_every)
        if cache:
            T.save(gen.state_dict(), ckpt)
            with open(f'etc/TimeVAE/{self.args.loc}/logs/{stem}-train_val.pkl', "wb") as f:
                pickle.dump(train_val_log, f)
            with open(f'etc/TimeVAE/{self.args.loc}/logs/{stem}-config.json', "w") as f:
                json.dump(cfg, f, indent=2, default=str)
            log(INFO, f"Generative TimeVAE saved on {ckpt}")
        return gen

    def _context_indices(self, n: int, n_ctx: int) -> T.Tensor:
        """`n` índices de janelas de contexto, percorrendo uma permutação embaralhada.

        Repetir a permutação (em vez de sortear com reposição) garante que todo contexto real
        seja usado antes de qualquer um repetir — com reposição e n=n_ctx, ~37% dos contextos
        nunca apareceriam. Semeado por `args.seed`, então é reprodutível."""
        perm = T.randperm(n_ctx, generator=T.Generator().manual_seed(self.args.seed))
        return perm.repeat(-(-n // n_ctx))[:n]

    def _prior_sample(self, gen, n: int, latent_dim: int, cons_idx: int, context=None):
        """Amostra n janelas do prior de um gerador dado e separa por tempo.

        Gerador condicional: `context` (janelas reais, (N, seq_len, feat)) é obrigatório — cada
        amostra recebe o clima/calendário de uma janela real e o decoder produz os canais
        endógenos coerentes com ele; a janela devolvida é o contexto com esses canais
        sobrescritos. Assim `X_syn` tem os mesmos `feat_dim` canais dos braços não-condicionais,
        e as features exógenas continuam correlacionadas com o alvo."""
        gen.eval(); gen.to(self.args.device)
        conditional = getattr(gen, "conditional", False)
        if conditional:
            if context is None or len(context) == 0:
                raise ValueError(
                    f"Client {self.cid}: a conditional generator needs real windows to draw the "
                    f"exogenous context from, but none were passed to _prior_sample."
                )
            if not T.is_tensor(context):
                context = T.from_numpy(np.asarray(context)).float()
            ctx_idx = self._context_indices(n, len(context))

        lags = self.args.num_lags
        X_parts, y_parts = [], []
        offset, remaining, bs = 0, n, 1024
        with T.no_grad():
            while remaining > 0:
                b = min(bs, remaining)
                z = T.randn(b, latent_dim, device=self.args.device)
                if conditional:
                    W_ctx = context[ctx_idx[offset:offset + b]].to(self.args.device)
                    W_syn = gen.assemble(W_ctx, gen.decoder(z, gen.condition(W_ctx)))
                else:
                    W_syn = gen.decoder(z)                              # (b, lags+H, feat)
                X_parts.append(W_syn[:, :lags, :].unsqueeze(-1).cpu())  # (b, lags, feat, 1)
                y_parts.append(W_syn[:, lags:, cons_idx].cpu())         # (b, H)
                offset += b
                remaining -= b
        return T.cat(X_parts, dim=0), T.cat(y_parts, dim=0)

    def sample_synthetic(self, n: Optional[int]=None, latent_dim: int=8, epochs: int=8, split: str="train",
                         gen=None, **gen_kwargs):
        """Amostra n janelas sintéticas do prior (z ~ N(0,I)) e separa por tempo:
        X_syn (n, lags, feat, 1) e y_syn (n, output_dim). n default = nº de janelas reais.
        Se `gen` for dado, usa esse gerador; senão treina/carrega via train_generative_timevae
        (repassando gen_kwargs: lr/beta/kl_anneal_frac/reconstruction_wt/tag/...)."""
        cons_idx = self._derive_consumption_index()
        if gen is None:
            gen = self.train_generative_timevae(latent_dim=latent_dim, epochs=epochs, split=split, **gen_kwargs)
        # As janelas reais do split servem para dimensionar a amostra e, no modo condicional,
        # como fonte do contexto exógeno.
        W = None
        if n is None or getattr(gen, "conditional", False):
            W, _ = self.build_generative_windows(split)
            n = len(W) if n is None else n
        context = None if W is None else T.from_numpy(W).float()
        X_syn, y_syn = self._prior_sample(gen, n, latent_dim, cons_idx, context=context)
        log(INFO, f"{self.cid}: sampled {n} synthetic windows from prior (split={split}) -> "
                  f"X_syn {tuple(X_syn.shape)}, y_syn {tuple(y_syn.shape)}")
        return X_syn, y_syn

    def _tstr_val_score(self, gen, latent_dim: int, cons_idx: int, model_name: str,
                        r_epochs: int, n_synthetic: int):
        """Objetivo da HPO: treina um regressor rápido em dados SINTÉTICOS (prior do `gen`) e
        avalia no conjunto de validação REAL. Retorna o MSE de validação (menor = melhor).
        Penaliza tanto reconstrução ruim quanto prior degenerado (o regressor treina no prior).

        O `best_model` é escolhido num split interno do SINTÉTICO, não na validação real: antes,
        selecionar e pontuar no mesmo conjunto tornava o objetivo otimista e favorecia trials que
        casavam com aquela validação específica."""
        from src.utils.functions import get_model
        context = None
        if getattr(gen, "conditional", False):
            W_ctx, _ = self.build_generative_windows("train")
            context = T.from_numpy(W_ctx).float()
        X_syn, y_syn = self._prior_sample(gen, n_synthetic, latent_dim, cons_idx, context=context)
        bs = getattr(self.args, "r_batch_size", self.args.batch_size)
        n_inner = max(1, int(0.2 * len(X_syn)))
        perm = T.randperm(len(X_syn), generator=T.Generator().manual_seed(self.args.seed))
        tr_idx, inner_idx = perm[n_inner:], perm[:n_inner]
        tr_loader = DataLoader(TensorDataset(X_syn[tr_idx], y_syn[tr_idx]), batch_size=bs, shuffle=True,
                               num_workers=self.args.num_workers)
        inner_val = DataLoader(TensorDataset(X_syn[inner_idx], y_syn[inner_idx]), batch_size=bs, shuffle=False,
                               num_workers=self.args.num_workers)
        real_val = DataLoader(self.val_dataset, batch_size=bs, shuffle=False, num_workers=self.args.num_workers)
        model = get_model(device=self.args.device, model=model_name, input_dim=self.input_dim,
                          out_dim=self.output_dim, lags=self.args.num_lags)
        best_model, _, _ = self.train(train_loader=tr_loader, val_loader=inner_val, model=model,
                                      epochs=r_epochs, optimizer="adamw", lr=1e-3, criterion="mse",
                                      early_stopping=False, patience=50, device=self.args.device)
        _, mse, *_ = self.test(best_model, real_val, nn.MSELoss(), device=self.args.device)
        del model, best_model, tr_loader, inner_val, real_val
        gc.collect()
        return float(mse)

    def optimize_generative_timevae(self, split: str="train", n_trials: int=15, hpo_epochs: int=40,
                                    hpo_r_epochs: int=8, hpo_n_synthetic: int=5000, model_name: str="lstm",
                                    full_epochs: int=200, seed: int=0):
        """Busca de hiperparâmetros (Optuna, um estudo por cliente) do TimeVAE gerador.
        Espaço: latent_dim, lr, reconstruction_wt, beta (β-VAE), kl_anneal_frac (annealing), a
        FORMA da loss de reconstrução — mean_time_wt e mean_feat_wt (os dois termos de média,
        antes acoplados num mean_axis_wt só), var_wt e grad_wt — e `cons_wt`, o peso do canal de
        consumo (sem ele o consumo fica com ~1% da loss). Objetivo: MSE de validação real de um
        regressor treinado no sintético (estilo TSTR). Ao final, retreina o melhor config por
        `full_epochs`, cacheia com tag '-opt' e devolve (best_params, best_value)."""
        import optuna
        mkdir_if_not_exists("optuna_db/")
        mkdir_if_not_exists(f'etc/TimeVAE/{self.args.loc}/logs/')
        storage = f"sqlite:///optuna_db/timevae_generation_{self.args.loc}.db"
        study = optuna.create_study(study_name=f"gen_study_{self.cid}_{split}", storage=storage,
                                    direction="minimize", load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        cons_idx = self._derive_consumption_index()

        def _gen_from_params(p, epochs, **kw):
            """Instancia o gerador a partir de um dict de hiperparâmetros do estudo."""
            return self.train_generative_timevae(
                latent_dim=p["latent_dim"], epochs=epochs, split=split, lr=p["lr"], beta=p["beta"],
                kl_anneal_frac=p["kl_anneal_frac"], reconstruction_wt=p["reconstruction_wt"],
                mean_time_wt=p["mean_time_wt"], mean_feat_wt=p["mean_feat_wt"],
                var_wt=p["var_wt"], grad_wt=p["grad_wt"],
                channel_wt=self.build_channel_wt({"consumption": p["cons_wt"]}), **kw)

        def objective(trial):
            hp = {
                "latent_dim": trial.suggest_categorical("latent_dim", [8, 16, 32, 64]),
                "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                "reconstruction_wt": trial.suggest_float("reconstruction_wt", 1.0, 12.0),
                "beta": trial.suggest_float("beta", 0.02, 2.0, log=True),
                "kl_anneal_frac": trial.suggest_float("kl_anneal_frac", 0.0, 0.7),
                # forma da loss de reconstrução: os dois termos de média agora são independentes
                # (a referência ativa só o de feature); mean_time_wt=0 remove o viés de achatamento
                "mean_time_wt": trial.suggest_float("mean_time_wt", 0.0, 1.0),
                "mean_feat_wt": trial.suggest_float("mean_feat_wt", 0.0, 1.0),
                "var_wt": trial.suggest_float("var_wt", 0.0, 30.0),
                "grad_wt": trial.suggest_float("grad_wt", 0.0, 10.0),
                # peso do canal de consumo na loss (1.0 = uniforme, como antes)
                "cons_wt": trial.suggest_float("cons_wt", 1.0, 50.0, log=True),
            }
            log(INFO, f"[GenHPO] {self.cid} trial {trial.number}/{n_trials}: {hp}")
            # log_every=hpo_epochs deixa ~2 linhas por trial: com 15 trials x 40 épocas o
            # andamento detalhado afogaria o log da busca.
            gen = _gen_from_params(hp, hpo_epochs, cache=False, log_every=hpo_epochs)
            score = self._tstr_val_score(gen, hp["latent_dim"], cons_idx, model_name, hpo_r_epochs, hpo_n_synthetic)
            del gen
            if T.cuda.is_available():
                T.cuda.empty_cache()
            gc.collect()
            log(INFO, f"[GenHPO] {self.cid} trial {trial.number} -> real-val MSE {score:.6f}")
            return score

        study.optimize(objective, n_trials=n_trials)
        best = study.best_params
        log(INFO, f"[GenHPO] {self.cid} best real-val MSE {study.best_value:.6f} with {best}")

        # Retreina o melhor config por full_epochs e cacheia com tag '-opt'.
        _gen_from_params(best, full_epochs, cache=True, force=True, tag="-opt")
        with open(f'etc/TimeVAE/{self.args.loc}/logs/{self.cid}-gen-{split}-opt-best_params.json', "w") as f:
            json.dump({"best_params": best, "best_value": study.best_value, "n_trials": n_trials}, f, indent=2)
        return best, study.best_value

    def fit_timevae(self, timevae: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int=8):
        timevae, train_val_log = self.train_timevae(timevae, train_loader, val_loader, epochs)
        return timevae, train_val_log

    @staticmethod
    def _resolve_mean_wts(mean_axis_wt, mean_time_wt, mean_feat_wt):
        """Compatibilidade: `mean_axis_wt` era um peso único aplicado aos DOIS termos de média.
        Quando os pesos separados não são dados, ele preenche os dois (comportamento antigo)."""
        if mean_time_wt is None:
            mean_time_wt = mean_axis_wt
        if mean_feat_wt is None:
            mean_feat_wt = mean_axis_wt
        return mean_time_wt, mean_feat_wt

    def _get_reconstruction_loss(self, X, X_recons, mean_axis_wt: float=1.0,
                                 var_wt: float=0.0, grad_wt: float=0.0,
                                 mean_time_wt: float=None, mean_feat_wt: float=None,
                                 channel_wt: Optional[T.Tensor]=None):
        """Loss de reconstrução configurável para o TimeVAE. X, X_recons: (batch, time, feat).
          - base: MSE por elemento (sempre presente).
          - mean_feat_wt: casa a média sobre as FEATURES a cada instante (reduz dim=2, preserva o
            eixo do tempo). É o único termo de média ATIVO na implementação de referência.
          - mean_time_wt: casa a média sobre o TEMPO por feature (reduz dim=1). A referência deixa
            este termo COMENTADO — uma reconstrução achatada no nível certo o satisfaz
            perfeitamente, então ele compete com a fidelidade ponto a ponto.
          - mean_axis_wt: peso legado aplicado aos dois acima quando os separados são None.
          - var_wt: casa o DESVIO-PADRÃO ao longo do tempo (por feature) — impede a reconstrução
            de colapsar para uma linha achatada de baixa variância.
          - grad_wt: casa a PRIMEIRA DIFERENÇA temporal — recompensa transições/picos afiados,
            atacando diretamente a perda dos picos de consumo.
          - channel_wt: pesos por canal, shape (feat_dim,), broadcast sobre (batch, time, feat).
            Sem ele a soma é uniforme sobre os 44 canais e o consumo fica com ~1% da loss (o resto
            vai para clima e para canais determinísticos como `minute`). None = uniforme.
        """
        mean_time_wt, mean_feat_wt = self._resolve_mean_wts(mean_axis_wt, mean_time_wt, mean_feat_wt)

        if channel_wt is None:
            w = None
        else:
            w = channel_wt.to(X.device, X.dtype).view(1, 1, -1)

        def _sq(diff):
            """Soma dos quadrados, ponderada por canal quando `channel_wt` é dado."""
            sq = T.pow(diff, 2)
            return T.sum(sq if w is None else sq * w)

        # base: MSE por elemento
        reconst_loss = _sq(X - X_recons)

        # NOTA: a ordem dos dois termos abaixo (tempo antes de feature) reproduz o laço
        # `for dim in (1, 2)` anterior, para que o caminho de seleção — que usa os defaults —
        # continue somando na mesma ordem e dando exatamente o mesmo float.

        # casamento da média sobre o tempo por feature (termo comentado na referência)
        if mean_time_wt > 0:
            reconst_loss = reconst_loss + mean_time_wt * _sq(T.mean(X, dim=1) - T.mean(X_recons, dim=1))

        # casamento da média sobre as features a cada instante (termo ativo na referência).
        # Reduz o eixo dos canais, então o peso por canal entra ANTES da média.
        if mean_feat_wt > 0:
            if w is None:
                mx, mr = T.mean(X, dim=2), T.mean(X_recons, dim=2)
            else:
                mx, mr = T.mean(X * w, dim=2), T.mean(X_recons * w, dim=2)
            reconst_loss = reconst_loss + mean_feat_wt * T.sum(T.pow(mx - mr, 2))

        # casamento de variância ao longo do tempo (combate reconstrução achatada)
        if var_wt > 0:
            reconst_loss = reconst_loss + var_wt * _sq(T.std(X, dim=1) - T.std(X_recons, dim=1))

        # casamento do gradiente temporal (primeira diferença) -> picos/bordas
        if grad_wt > 0:
            dX = X[:, 1:, :] - X[:, :-1, :]
            dXr = X_recons[:, 1:, :] - X_recons[:, :-1, :]
            reconst_loss = reconst_loss + grad_wt * _sq(dX - dXr)

        return reconst_loss

    def _vae_reconstruct(self, model, X):
        """Um passo encoder->decoder, tratando modelo condicional e não-condicional.

        Devolve `(target, reconstruction, z_mean, z_log_var)`. No modo condicional o alvo da
        reconstrução é só a fatia dos canais gerados — o decoder não produz os exógenos — e o
        decoder recebe o embedding do contexto. No modo comum o caminho é literalmente o de
        antes (`model.decoder(z)` sobre `X` inteiro), então o VAE de seleção não muda."""
        z_mean, z_log_var, z = model.encoder(X)
        if getattr(model, "conditional", False):
            return model.target(X), model.decoder(z, model.condition(X)), z_mean, z_log_var
        return X, model.decoder(z), z_mean, z_log_var

    def loss_function(self, X, X_recons, z_mean, z_log_var, beta: float=1.0, reconstruction_wt: float=None,
                      mean_axis_wt: float=1.0, var_wt: float=0.0, grad_wt: float=0.0,
                      mean_time_wt: float=None, mean_feat_wt: float=None,
                      channel_wt: Optional[T.Tensor]=None):
        """VAE loss = reconstruction_wt * recon + beta * KL.
        beta escala o termo KL (β-VAE / KL annealing); reconstruction_wt sobrepõe o default de args.
        Os demais configuram a loss de reconstrução (ver _get_reconstruction_loss).
        Defaults (beta=1.0, reconstruction_wt=None->args, mean_axis_wt=1.0, mean_time_wt/
        mean_feat_wt=None, var_wt=0, grad_wt=0, channel_wt=None) reproduzem exatamente o
        comportamento anterior, usado pelo TimeVAE de seleção."""
        if reconstruction_wt is None:
            reconstruction_wt = self.args.reconstruction_wt
        reconstruction_loss = self._get_reconstruction_loss(X, X_recons, mean_axis_wt=mean_axis_wt,
                                                             var_wt=var_wt, grad_wt=grad_wt,
                                                             mean_time_wt=mean_time_wt, mean_feat_wt=mean_feat_wt,
                                                             channel_wt=channel_wt)
        kl_loss = -0.5 * T.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        total_loss = reconstruction_wt * reconstruction_loss + beta * kl_loss
        return total_loss, reconstruction_loss, kl_loss


    def train_timevae(self, model: nn.Module, train_loader, val_loader, max_epochs,
                      lr: float=1e-3, beta: float=1.0, kl_anneal_epochs: int=0, reconstruction_wt: float=None,
                      mean_axis_wt: float=1.0, var_wt: float=0.0, grad_wt: float=0.0,
                      mean_time_wt: float=None, mean_feat_wt: float=None,
                      channel_wt: Optional[T.Tensor]=None, log_every: Optional[int]=None):
        """Treina um TimeVAE. Parâmetros generativos configuráveis:
          - lr: taxa do AdamW (antes fixo em 1e-3).
          - beta: peso final do termo KL (β-VAE).
          - kl_anneal_epochs: nº de épocas para subir o KL de 0 até beta (annealing);
            0 = sem annealing (beta constante). Evita o colapso/espalhamento do prior.
          - reconstruction_wt: sobrepõe args.reconstruction_wt quando fornecido.
          - mean_axis_wt/mean_time_wt/mean_feat_wt/var_wt/grad_wt/channel_wt: forma da loss de
            reconstrução (ver _get_reconstruction_loss); os defaults reproduzem o comportamento
            original do VAE de seleção.
          - log_every: intervalo (em épocas) das linhas de andamento; None = ~25 linhas no total.
        O best_model é escolhido pela val loss avaliada com o beta FINAL (comparação justa
        entre épocas, independente do estágio de annealing)."""
        loss_kwargs = dict(reconstruction_wt=reconstruction_wt, mean_axis_wt=mean_axis_wt,
                           mean_time_wt=mean_time_wt, mean_feat_wt=mean_feat_wt,
                           var_wt=var_wt, grad_wt=grad_wt, channel_wt=channel_wt)
        timevae_loss_dict = defaultdict(list)
        optimizer = T.optim.AdamW(model.parameters(), lr=lr)
        scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
        best_val_loss = np.inf
        best_model = None

        # O tqdm escreve com '\r' no stderr: num terminal fica ótimo, mas em `docker compose logs`
        # (sem TTY) vira uma linha gigante ou nada. Fora de TTY ele é desligado e o andamento sai
        # pelas linhas de log abaixo, uma por época amostrada.
        interactive = sys.stderr.isatty()
        if log_every is None:
            log_every = max(1, max_epochs // 25)  # ~25 linhas por treino, independente do total
        n_train, n_val = len(train_loader.dataset), len(val_loader.dataset)
        log(INFO, f"{self.cid}: TimeVAE training started — {max_epochs} epochs, {n_train} train / "
                  f"{n_val} val windows, {len(train_loader)} batches/epoch, lr={lr}, beta={beta}"
                  f"{f' (KL annealing over {kl_anneal_epochs} epochs)' if kl_anneal_epochs > 0 else ''}, "
                  f"logging every {log_every} epoch(s)")
        started = time.time()

        with tqdm(total=max_epochs, desc=f"Training {self.cid}'s TimeVAE model",
                  disable=not interactive) as pbar:
            for epoch in range(max_epochs):
                # KL annealing: beta_t sobe linearmente de 0 até beta ao longo de kl_anneal_epochs.
                beta_t = beta if kl_anneal_epochs <= 0 else beta * min(1.0, (epoch + 1) / kl_anneal_epochs)
                model.train()
                model.to(self.args.device)
                total_loss = []
                reconstruction_loss = []
                kl_loss = []

                for X, y in train_loader:
                    X, y = X.to(self.args.device), y.to(self.args.device)
                    X = X.reshape(X.size(0), X.size(1), X.size(2))
                    optimizer.zero_grad()
                    target, reconstruction, z_mean, z_log_var = self._vae_reconstruct(model, X)
                    loss, recon_loss, kl = self.loss_function(target, reconstruction, z_mean, z_log_var,
                                                              beta=beta_t, **loss_kwargs)
                    loss = loss / X.size(0)
                    recon_loss = recon_loss / X.size(0)
                    kl = kl / X.size(0)

                    loss.backward()
                    optimizer.step()

                    total_loss.append(loss.item())
                    reconstruction_loss.append(recon_loss.item())
                    kl_loss.append(kl.item())

                # Val loss com beta FINAL (comparação justa entre épocas).
                val_loss, val_reconstruction_loss, val_kl_loss = self.test_timevae(
                    model, val_loader, beta=beta, **loss_kwargs)
                timevae_loss_dict["train_total_loss"].append(np.mean(total_loss))
                timevae_loss_dict["train_reconstruction_loss"].append(np.mean(reconstruction_loss))
                timevae_loss_dict["train_kl_loss"].append(np.mean(kl_loss))
                timevae_loss_dict["val_total_loss"].append(val_loss)
                timevae_loss_dict["val_reconstruction_loss"].append(val_reconstruction_loss)
                timevae_loss_dict["val_kl_loss"].append(val_kl_loss)

                scheduler.step(val_loss)
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(model)
                pbar.set_postfix({'epoch': epoch + 1, 'beta': round(beta_t, 3),
                                  'loss': round(val_loss, 4),
                                  'recon_loss': round(val_reconstruction_loss, 4),
                                  'kl_loss': round(val_kl_loss, 4)})
                pbar.update()

                # Linha de andamento: primeira época, última, e a cada `log_every`. A melhoria do
                # best não dispara linha (seriam ~200 numa rodada longa e ~600 durante a HPO) —
                # ela aparece como '*' e o best corrente está em toda linha. ETA pelo tempo médio.
                done = epoch + 1
                if done == 1 or done == max_epochs or done % log_every == 0:
                    elapsed = time.time() - started
                    eta = elapsed / done * (max_epochs - done)
                    log(INFO,
                        f"{self.cid}: TimeVAE epoch {done}/{max_epochs} "
                        f"({100 * done / max_epochs:5.1f}%) | "
                        f"train {np.mean(total_loss):10.3f} (recon {np.mean(reconstruction_loss):9.3f}, "
                        f"kl {np.mean(kl_loss):7.3f}) | val {val_loss:10.3f} "
                        f"(recon {val_reconstruction_loss:9.3f}, kl {val_kl_loss:7.3f}) | "
                        f"best {best_val_loss:10.3f}{' *' if improved else '  '} | "
                        f"beta {beta_t:.3f} | lr {optimizer.param_groups[0]['lr']:.2e} | "
                        f"{elapsed / done:.1f}s/epoch | elapsed {_fmt_dur(elapsed)}, ETA {_fmt_dur(eta)}")

        log(INFO, f"{self.cid}: TimeVAE training finished in {_fmt_dur(time.time() - started)} — "
                  f"best val loss {best_val_loss:.4f}")
        # Se, por algum motivo, nenhuma época melhorou o best (não deve ocorrer), cai no modelo atual.
        return best_model if best_model is not None else model, timevae_loss_dict

    def test_timevae(self, model, val_loader, beta: float=1.0, reconstruction_wt: float=None,
                     mean_axis_wt: float=1.0, var_wt: float=0.0, grad_wt: float=0.0,
                     mean_time_wt: float=None, mean_feat_wt: float=None,
                     channel_wt: Optional[T.Tensor]=None):
        model.to(self.args.device)
        model.eval()

        total_loss = []
        reconstruction_loss = []
        kl_loss = []

        with T.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.args.device), y.to(self.args.device)
                X = X.reshape(X.size(0), X.size(1), X.size(2))
                target, reconstruction, z_mean, z_log_var = self._vae_reconstruct(model, X)
                loss, recon_loss, kl = self.loss_function(target, reconstruction, z_mean, z_log_var,
                                                          beta=beta, reconstruction_wt=reconstruction_wt,
                                                          mean_axis_wt=mean_axis_wt, var_wt=var_wt, grad_wt=grad_wt,
                                                          mean_time_wt=mean_time_wt, mean_feat_wt=mean_feat_wt,
                                                          channel_wt=channel_wt)
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
