import torch as T
import torch.nn as nn
from src.models.timeVAE.inner_layers import Sampling, LevelModel, ResidualConnection, TrendLayer, SeasonalLayer

class TimeVAEEncoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_sizes, latent_dim, device):
        super(TimeVAEEncoder, self).__init__()
        
        self.device = device
        self.layers = []
        self.layers.append(nn.Conv1d(feat_dim, hidden_sizes[0], kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU())

        for i, num_filters in enumerate(hidden_sizes[1:]):
            self.layers.append(nn.Conv1d(hidden_sizes[i], num_filters, kernel_size=3, stride=2, padding=1))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Flatten())

        self.encoder_last_dense_dim = self._get_last_dense_dim(seq_len, feat_dim, hidden_sizes)

        self.encoder = nn.Sequential(*self.layers)
        self.z_mean = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.z_log_var = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.to(device)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = Sampling(self.device)([z_mean, z_log_var])
        return z_mean, z_log_var, z

    def _get_last_dense_dim(self, seq_len, feat_dim, hidden_layer_sizes):
        with T.no_grad():
            x = T.randn(1, feat_dim, seq_len)
            for conv in self.layers:
                x = conv(x)
            return x.numel()


class TimeVAEDecoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim, device, trend_poly=0, custom_seas=None, use_residual_conn=True, encoder_last_dense_dim=None, cond_dim=0):
        super(TimeVAEDecoder, self).__init__()
        self.seq_len=seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        self.encoder_last_dense_dim = encoder_last_dense_dim
        self.device = device

        # Condicionamento: toda sub-camada do decoder consome um vetor (B, D) e devolve
        # (B, seq_len, feat_dim). Concatenar o embedding do contexto a `z` e construir as
        # sub-camadas com D = latent_dim + cond_dim condiciona o decoder inteiro sem alterar
        # nenhuma delas. cond_dim=0 -> D = latent_dim, exatamente o modelo não-condicional.
        self.cond_dim = cond_dim
        dec_dim = latent_dim + cond_dim

        self.level_model = LevelModel(dec_dim, self.feat_dim, self.seq_len, device)

        # As camadas de tendência e sazonalidade são construídas AQUI, não no forward. Instanciá-las
        # dentro do forward (como fazia a implementação de referência) criava pesos novos e
        # aleatórios a cada chamada: não eram registrados como parâmetros, nunca eram treinados nem
        # salvos no state_dict, e o decoder devolvia ruído sempre que trend_poly>0 ou custom_seas
        # estavam ativos. Com os defaults (trend_poly=0, custom_seas=None) o bug ficava latente e o
        # modelo era, na prática, só level+residual — um VAE convolucional comum, não o TimeVAE.
        self.trend_layer = None
        if trend_poly is not None and trend_poly > 0:
            self.trend_layer = TrendLayer(seq_len, feat_dim, dec_dim, trend_poly, device)

        self.seasonal_layer = None
        if custom_seas is not None and len(custom_seas) > 0:
            self.seasonal_layer = SeasonalLayer(seq_len, feat_dim, dec_dim, custom_seas, device)

        if use_residual_conn:
            self.residual_conn = ResidualConnection(seq_len, feat_dim, hidden_layer_sizes, dec_dim, encoder_last_dense_dim, device)

        self.to(device)

    def forward(self, z, c=None):
        """z: (B, latent_dim); c: (B, cond_dim) obrigatório quando cond_dim > 0."""
        if self.cond_dim > 0:
            if c is None:
                raise ValueError("Conditional decoder (cond_dim > 0) called without a condition vector.")
            z = T.cat([z, c], dim=1)
        elif c is not None:
            raise ValueError("Condition vector passed to a non-conditional decoder (cond_dim == 0).")

        outputs = self.level_model(z)
        if self.trend_layer is not None:
            outputs = outputs + self.trend_layer(z)

        # custom seasons
        if self.seasonal_layer is not None:
            outputs = outputs + self.seasonal_layer(z)

        if self.use_residual_conn:
            outputs = outputs + self.residual_conn(z)

        return outputs
