import torch.nn as nn
from src.models.timeVAE.inner_layers import Sampling
from src.models.timeVAE.encoder_decoder import TimeVAEEncoder, TimeVAEDecoder


class TimeVAE(nn.Module):
    def __init__(self, hidden_sizes, trend_poly, custom_seats, use_residual_conn, seq_len, feat_dim, latent_dim, device):
        super(TimeVAE, self).__init__()

        self.sampling = Sampling(device)
        self.encoder = TimeVAEEncoder(seq_len, feat_dim, hidden_sizes, latent_dim, device)
        self.decoder = TimeVAEDecoder(seq_len, feat_dim, hidden_sizes, latent_dim, device, trend_poly=trend_poly, custom_seas=custom_seats, use_residual_conn=use_residual_conn, encoder_last_dense_dim=self.encoder.encoder_last_dense_dim)
        
        self._init_weights()


    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


    def forward(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        return z_mean, z_log_var, z, x_decoded
