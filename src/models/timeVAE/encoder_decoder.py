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
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim, device, trend_poly=0, custom_seas=None, use_residual_conn=True, encoder_last_dense_dim=None):
        super(TimeVAEDecoder, self).__init__()
        self.seq_len=seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        self.encoder_last_dense_dim = encoder_last_dense_dim
        self.device = device
        self.level_model = LevelModel(self.latent_dim, self.feat_dim, self.seq_len, device)

        if use_residual_conn:
            self.residual_conn = ResidualConnection(seq_len, feat_dim, hidden_layer_sizes, latent_dim, encoder_last_dense_dim, device)

        self.to(device)

    def forward(self, z):
        outputs = self.level_model(z)
        if self.trend_poly is not None and self.trend_poly > 0:
            trend_vals = TrendLayer(self.seq_len, self.feat_dim, self.latent_dim, self.trend_poly, self.device)(z)
            outputs += trend_vals

        # custom seasons
        if self.custom_seas is not None and len(self.custom_seas) > 0:
            cust_seas_vals = SeasonalLayer(self.seq_len, self.feat_dim, self.latent_dim, self.custom_seas, self.device)(z)
            outputs += cust_seas_vals

        if self.use_residual_conn:
            residuals = self.residual_conn(z)
            outputs += residuals

        return outputs
