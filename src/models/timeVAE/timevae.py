import torch as T
import torch.nn as nn
from src.models.timeVAE.inner_layers import Sampling, ConditionEncoder
from src.models.timeVAE.encoder_decoder import TimeVAEEncoder, TimeVAEDecoder


class TimeVAE(nn.Module):
    """VAE de séries temporais, com modo condicional opcional.

    Não-condicional (`gen_idx=None`, o default): encoder e decoder operam sobre todos os
    `feat_dim` canais. É o modelo usado pela seleção de clientes e pelos braços A0-A5 — o
    caminho é idêntico ao de antes, sem parâmetro novo no state_dict.

    Condicional (`gen_idx` dado): o decoder MODELA apenas os canais de `gen_idx` (os endógenos:
    consumo, geração, tensões e derivados) e é CONDICIONADO nos demais (`cond_idx`: clima e
    calendário), que são exógenos ou determinísticos e não fazem sentido inventar. O encoder
    continua vendo a janela inteira — q(z|x,c). Na amostragem, o contexto vem de uma janela real
    e o consumo gerado sai coerente com ele.
    """

    def __init__(self, hidden_sizes, trend_poly, custom_seats, use_residual_conn, seq_len,
                 feat_dim, latent_dim, device, gen_idx=None, cond_idx=None, cond_dim=0):
        super(TimeVAE, self).__init__()

        self.conditional = gen_idx is not None
        self.sampling = Sampling(device)
        self.encoder = TimeVAEEncoder(seq_len, feat_dim, hidden_sizes, latent_dim, device)

        if self.conditional:
            if not cond_idx:
                raise ValueError("Conditional TimeVAE needs a non-empty cond_idx (the exogenous channels).")
            if cond_dim <= 0:
                raise ValueError(f"Conditional TimeVAE needs cond_dim > 0, got {cond_dim}.")
            # Índices como buffers: acompanham .to(device) e ficam gravados no state_dict, então
            # um checkpoint condicional carrega sabendo qual partição de canais o gerou.
            self.register_buffer("gen_idx", T.as_tensor(list(gen_idx), dtype=T.long))
            self.register_buffer("cond_idx", T.as_tensor(list(cond_idx), dtype=T.long))
            self.cond_encoder = ConditionEncoder(seq_len, len(cond_idx), hidden_sizes, cond_dim, device)
            dec_feat_dim = len(gen_idx)
        else:
            dec_feat_dim, cond_dim = feat_dim, 0

        self.decoder = TimeVAEDecoder(seq_len, dec_feat_dim, hidden_sizes, latent_dim, device,
                                      trend_poly=trend_poly, custom_seas=custom_seats,
                                      use_residual_conn=use_residual_conn,
                                      encoder_last_dense_dim=self.encoder.encoder_last_dense_dim,
                                      cond_dim=cond_dim)

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def condition(self, X):
        """Embedding do contexto exógeno de X (janela completa). None se não-condicional."""
        return self.cond_encoder(X[:, :, self.cond_idx]) if self.conditional else None

    def target(self, X):
        """A parte de X que o decoder reconstrói: os canais gerados, ou X inteiro."""
        return X[:, :, self.gen_idx] if self.conditional else X

    def assemble(self, X_context, generated):
        """Remonta janelas completas: o contexto real com os canais gerados sobrescritos.

        Mantém a saída com os `feat_dim` canais originais, para que tudo a jusante (regressor,
        gráficos, métricas) continue enxergando o mesmo formato dos braços não-condicionais."""
        if not self.conditional:
            return generated
        out = X_context.clone()
        out[:, :, self.gen_idx] = generated
        return out

    def forward(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        x_decoded = self.decoder(z_mean, self.condition(X))
        return z_mean, z_log_var, z, x_decoded
