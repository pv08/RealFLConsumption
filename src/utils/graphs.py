"""Gráficos de diagnóstico do TimeVAE **generativo** (ver eval_timevae.py).

Este módulo responde à pergunta "o gerador está capturando o quê?" comparando, no mesmo
cliente e no mesmo split, dois geradores treinados de formas diferentes (ex.: loss original
vs loss estrutural; baseline vs otimizado por HPO). São quatro painéis:

  (A) curva de consumo em ordem temporal: real vs **reconstruído** (encoder -> z_mean -> decoder),
      com MSE e correlação de Pearson na legenda -> fidelidade ponto a ponto;
  (B) histograma de densidade do consumo-alvo: real vs **amostras do prior** (z ~ N(0,I))
      -> fidelidade da distribuição marginal (é o que o TSTR de fato consome);
  (C) perfil temporal médio ±1σ ao longo da janela: real vs prior -> fidelidade do perfil diário;
  (D) janelas individuais: real vs reconstruído -> onde os picos se perdem.

Todos os valores são invertidos para kW com o `y_scaler` do cliente. Isso vale para a janela
inteira (não só para o alvo) porque `x_scaler` e `y_scaler` são ajustados sobre a MESMA coluna
`consumption` das MESMAS linhas de treino (ver Processing.make_preprocessing), então o
min/max do canal de consumo é idêntico nos dois.

Uso típico (com um `ClientLearning` já construído, como em eval_timevae.py):

    gen_a = cl.train_generative_timevae(latent_dim=8,  epochs=200, split="test")
    gen_b = cl.train_generative_timevae(latent_dim=32, epochs=200, split="test", tag="-opt")
    plot_baseline_vs_opt(cl, gen_a, gen_b, latent_dim_a=8, latent_dim_b=32, split="test")
"""
import os
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from logging import INFO
from typing import Optional, Sequence, Tuple

from src.utils.functions import mkdir_if_not_exists
from src.utils.logger import log

# Cores fixas: o real é sempre o preto/cinza, o gerador A o laranja e o B o verde.
COLOR_REAL = "black"
COLOR_A = "tab:orange"
COLOR_B = "tab:green"


# ----------------------------------------------------------------------
# Primitivas: reconstrução, amostragem do prior e conversão para kW
# ----------------------------------------------------------------------
def reconstruct_windows(cl, gen, W: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Reconstrói janelas reais pelo gerador de forma DETERMINÍSTICA: usa `z_mean` do encoder
    (não a amostra `z`), então duas chamadas dão o mesmo resultado. W: (N, seq_len, feat).

    Gerador condicional: o decoder devolve só os canais modelados, então a saída é remontada
    sobre a própria janela real — os exógenos ficam idênticos aos de entrada (é o que o modelo
    assume) e os endógenos são os reconstruídos. O shape de retorno é sempre o de W."""
    gen.eval()
    gen.to(cl.args.device)
    conditional = getattr(gen, "conditional", False)
    out = []
    with T.no_grad():
        for i in range(0, len(W), batch_size):
            X = T.from_numpy(W[i:i + batch_size]).float().to(cl.args.device)
            z_mean, _, _ = gen.encoder(X)
            dec = gen.decoder(z_mean, gen.condition(X)) if conditional else gen.decoder(z_mean)
            out.append((gen.assemble(X, dec) if conditional else dec).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty_like(W)


def prior_windows(cl, gen, n: int, latent_dim: int, batch_size: int = 1024,
                  context: Optional[np.ndarray] = None) -> np.ndarray:
    """Amostra n janelas COMPLETAS do prior (z ~ N(0,I) -> decoder). Diferente de
    `ClientLearning._prior_sample`, que já devolve X/y separados por tempo; aqui a janela
    inteira é preservada para os painéis de perfil e distribuição.

    Gerador condicional: `context` (janelas reais) é obrigatório — cada amostra recebe o
    clima/calendário de uma janela real e devolve a janela completa remontada."""
    gen.eval()
    gen.to(cl.args.device)
    conditional = getattr(gen, "conditional", False)
    if conditional:
        if context is None or len(context) == 0:
            raise ValueError("prior_windows: a conditional generator needs `context` (real windows).")
        ctx = T.from_numpy(np.asarray(context)).float()
        ctx_idx = cl._context_indices(n, len(ctx))
    out, offset, remaining = [], 0, n
    with T.no_grad():
        while remaining > 0:
            b = min(batch_size, remaining)
            z = T.randn(b, latent_dim, device=cl.args.device)
            if conditional:
                W_ctx = ctx[ctx_idx[offset:offset + b]].to(cl.args.device)
                W_syn = gen.assemble(W_ctx, gen.decoder(z, gen.condition(W_ctx)))
            else:
                W_syn = gen.decoder(z)
            out.append(W_syn.cpu().numpy())
            offset += b
            remaining -= b
    return np.concatenate(out, axis=0)


def to_kw(values, y_scaler) -> np.ndarray:
    """Inverte a escala do canal de consumo para kW, preservando o shape de entrada."""
    arr = np.asarray(values, dtype=np.float64)
    if y_scaler is None:
        return arr
    return y_scaler.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)


def target_series(W: np.ndarray, num_lags: int, cons_idx: int) -> np.ndarray:
    """Série do ALVO em ordem temporal: o primeiro passo previsto de cada janela.
    Com stride 1 entre janelas, isso reconstitui a própria série de consumo."""
    return W[:, num_lags, cons_idx]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Correlação de Pearson tolerante a série constante (devolve nan em vez de estourar)."""
    a, b = np.ravel(a), np.ravel(b)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------------------------
# Painéis
# ----------------------------------------------------------------------
def _panel_curve(ax, real, recons: Sequence[Tuple[str, np.ndarray, str]], xlabel: str):
    """(A) Curva do alvo em ordem temporal: real vs cada reconstrução, com MSE/r na legenda.
    `recons`: lista de (label, série, cor). Métricas em kW (mesma unidade do eixo)."""
    ax.plot(real, color=COLOR_REAL, lw=1.8, label="real (teste)")
    for label, series, color in recons:
        mse = float(np.mean((real - series) ** 2))
        ax.plot(series, color=color, ls="--", lw=1.2,
                label=f"{label} (MSE {mse:.3f}, r {_pearson(real, series):.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("consumo (kW)")
    ax.legend(loc="upper left", fontsize=8)


def _panel_dist(ax, real, priors: Sequence[Tuple[str, np.ndarray, str]], bins: int = 80):
    """(B) Densidade marginal do consumo-alvo: real vs amostras do prior de cada gerador."""
    ax.hist(np.ravel(real), bins=bins, density=True, alpha=.75, color="0.35", label="real (teste)")
    for label, samples, color in priors:
        ax.hist(np.ravel(samples), bins=bins, density=True, alpha=.55, color=color, label=label)
    ax.set_xlabel("consumo (kW)")
    ax.set_ylabel("densidade")
    ax.legend(loc="upper right", fontsize=8)


def _panel_profile(ax, real_win: np.ndarray, prior_win: np.ndarray, prior_label: str, color: str):
    """(C) Perfil temporal médio ±1σ ao longo da janela (real vs prior)."""
    for win, label, c in ((real_win, "real (teste) média", COLOR_REAL),
                          (prior_win, f"{prior_label} (prior) média", color)):
        mu, sd = win.mean(axis=0), win.std(axis=0)
        ax.plot(mu, color=c, lw=1.5, label=label)
        ax.fill_between(np.arange(len(mu)), mu - sd, mu + sd, color=c, alpha=.2)
    ax.set_xlabel("posição na janela")
    ax.set_ylabel("consumo (kW)")
    ax.legend(loc="upper left", fontsize=8)


def _panel_windows(ax, real_win: np.ndarray, recon_win: np.ndarray, idxs: Sequence[int], label: str):
    """(D) Janelas individuais: real (sólido) vs reconstruído (tracejado), uma cor por janela."""
    for k, i in enumerate(idxs):
        color = f"C{k}"
        ax.plot(real_win[i], color=color, lw=1.5, label=f"real win {i}")
        ax.plot(recon_win[i], color=color, ls="--", lw=1.2, label=f"{label} recon win {i}")
    ax.set_xlabel("passo (15 min)")
    ax.set_ylabel("consumo (kW)")
    ax.legend(loc="upper left", fontsize=8)


# ----------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------
def plot_generator_comparison(cl, gen_a, gen_b, latent_dim_a: int, latent_dim_b: int,
                              split: str = "test", label_a: str = "baseline", label_b: str = "otimizado",
                              title: Optional[str] = None, out_path: Optional[str] = None,
                              full: bool = True, n_prior: int = 5000,
                              window_idxs: Optional[Sequence[int]] = None, bins: int = 80,
                              batch_size: int = 256, seed: Optional[int] = None) -> Optional[str]:
    """Compara dois geradores no mesmo cliente/split e salva a figura.

    `full=True`  -> 2x2 (curva, distribuição, perfil médio, janelas individuais);
    `full=False` -> 1x2 (curva + distribuição), suficiente para ablação de loss.
    Os painéis (C) e (D) descrevem apenas o gerador B (o "novo"/otimizado).
    `n_prior` é o nº de janelas amostradas do prior de CADA gerador — precisa ser bem maior
    que o nº de janelas reais para o histograma (B) e o desvio (C) não ficarem ruidosos.
    `seed` fixa a amostragem do prior (a reconstrução já é determinística).
    Devolve o caminho salvo (ou None se o split não tem janelas).
    """
    if seed is not None:
        T.manual_seed(seed)

    W, cons_idx = cl.build_generative_windows(split)
    if len(W) == 0:
        log(INFO, f"{cl.cid}: split '{split}' has no generative windows; skipping plot")
        return None

    num_lags = cl.args.num_lags

    # Reconstruções (determinísticas) e amostras do prior de cada gerador.
    rec_a = reconstruct_windows(cl, gen_a, W, batch_size)
    rec_b = reconstruct_windows(cl, gen_b, W, batch_size)
    prior_a = prior_windows(cl, gen_a, n_prior, latent_dim_a, context=W)
    prior_b = prior_windows(cl, gen_b, n_prior, latent_dim_b, context=W)

    # Tudo em kW: séries do alvo (painéis A/B) e janelas inteiras do consumo (C/D).
    real_curve = to_kw(target_series(W, num_lags, cons_idx), cl.y_scaler)
    curve_a = to_kw(target_series(rec_a, num_lags, cons_idx), cl.y_scaler)
    curve_b = to_kw(target_series(rec_b, num_lags, cons_idx), cl.y_scaler)
    prior_curve_a = to_kw(target_series(prior_a, num_lags, cons_idx), cl.y_scaler)
    prior_curve_b = to_kw(target_series(prior_b, num_lags, cons_idx), cl.y_scaler)

    nrows, ncols = (2, 2) if full else (1, 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 4.5 * nrows))
    axes = np.ravel(axes)

    _panel_curve(axes[0], real_curve,
                 [(label_a, curve_a, COLOR_A), (label_b, curve_b, COLOR_B)],
                 xlabel="janela de teste (ordem temporal)" if full else "janela de teste")
    axes[0].set_title("(A) Curva de consumo do teste: real vs reconstruído")

    _panel_dist(axes[1], real_curve,
                [(f"prior {label_a}", prior_curve_a, COLOR_A),
                 (f"prior {label_b}", prior_curve_b, COLOR_B)], bins=bins)
    axes[1].set_title("(B) Distribuição do consumo (alvo)")

    if full:
        real_win = to_kw(W[:, :, cons_idx], cl.y_scaler)
        _panel_profile(axes[2], real_win, to_kw(prior_b[:, :, cons_idx], cl.y_scaler), label_b, COLOR_B)
        axes[2].set_title("(C) Perfil temporal médio ±1σ")

        idxs = list(window_idxs) if window_idxs is not None else [0, len(W) - 1]
        _panel_windows(axes[3], real_win, to_kw(rec_b[:, :, cons_idx], cl.y_scaler), idxs, label_b)
        axes[3].set_title(f"(D) Janelas de teste: real vs reconstruído ({label_b})")

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95) if title else None)

    out_path = out_path or f'etc/TimeVAE/{cl.args.loc}/results/plots/{cl.cid}_generator_comparison_{split}.png'
    outdir = os.path.dirname(out_path)
    if outdir:
        mkdir_if_not_exists(outdir)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log(INFO, f"{cl.cid}: generator comparison plot saved to {out_path}")
    return out_path


def _tsne_real_indices(n_windows: int, seq_len: int, max_samples: int, min_points: int = 50,
                       contiguous: bool = True):
    """Índices das janelas reais para o t-SNE: espalhados pela série e sem sobreposição.

    Pegar as `max_samples` primeiras janelas (o que esta função substitui) distorce o gráfico de
    duas formas. Com stride 1 janelas vizinhas compartilham `seq_len-1` passos, então os pontos
    reais formam uma cadeia quase contínua e o t-SNE a desenha como filamentos — enquanto as
    amostras do prior são i.i.d. e nunca poderiam formar fios, o que faz "não misturar" parecer
    falha do modelo quando é artefato da amostragem. E as primeiras janelas cobrem só o começo
    da série (~3% no split de treino), comparando um trecho curto de uma estação contra um prior
    que amostra o período inteiro.

    Por isso o passo é `max(cobertura, seq_len)`: cobre toda a extensão temporal e garante que
    duas janelas escolhidas não compartilhem nenhum passo. Em splits curtos (o de teste tem ~95
    janelas, ~1 dia) isso deixaria pouquíssimos pontos, então a sobreposição volta a ser aceita —
    o retorno sinaliza isso para o gráfico registrar no título.

    `contiguous=False` diz que `W` já foi subamostrado na construção (`--window_stride`,
    `--align_to_day`), caso em que entradas vizinhas não compartilham passos e o único critério
    que resta é cobrir a série.

    Retorna `(idx, overlapping)`.
    """
    coverage_stride = max(1, n_windows // max_samples)
    min_gap = seq_len if contiguous else 1
    idx = np.arange(0, n_windows, max(coverage_stride, min_gap))[:max_samples]
    if len(idx) >= min_points or not contiguous:
        return idx, False
    return np.arange(0, n_windows, coverage_stride)[:max_samples], True


def plot_tsne(cl, gen, latent_dim: int, split: str = "test", label: str = "gerado (prior)",
              channel: Optional[str] = "consumption", max_samples: int = 1000,
              perplexity: int = 40, out_path: Optional[str] = None, title: Optional[str] = None,
              seed: int = 42) -> Optional[str]:
    """t-SNE das janelas reais vs amostras do prior — o diagnóstico padrão do TimeVAE.

    Responde o que os painéis de curva não respondem: se as amostras do prior cobrem a mesma
    variedade que os dados reais, ou se caem numa região à parte. Portado de `visualize.py` da
    implementação de referência (`visualize_and_save_tsne`), com duas diferenças: a referência
    reduz cada janela pela média sobre os canais, o que com 44 canais dilui justamente o canal
    de interesse (aqui `channel="consumption"` é o default; passe `channel=None` para reproduzir
    a média), e as janelas reais são amostradas por `_tsne_real_indices` em vez de virem da
    cabeça da série.
    """
    from sklearn.manifold import TSNE

    W, cons_idx = cl.build_generative_windows(split)
    if len(W) == 0:
        log(INFO, f"{cl.cid}: split '{split}' has no generative windows; skipping t-SNE")
        return None

    # `W` é contíguo (stride 1) quando a janela i+1 é a i deslocada de um passo; se veio
    # subamostrado, vizinhos já não se sobrepõem e a restrição de passo mínimo não se aplica.
    contiguous = len(W) > 1 and np.allclose(W[0, 1:], W[1, :-1])
    real_idx, overlapping = _tsne_real_indices(len(W), W.shape[1], max_samples,
                                               contiguous=contiguous)
    n = len(real_idx)
    T.manual_seed(seed)
    prior = prior_windows(cl, gen, n, latent_dim, context=W)

    def _flatten(windows, idx):
        """(N, seq_len, feat) -> (n, seq_len): um canal, ou a média sobre os canais."""
        sel = windows[idx]
        arr = sel[:, :, cons_idx] if channel == "consumption" else (
            sel[:, :, cl.channel_index(channel)] if channel else sel.mean(axis=2))
        return to_kw(arr, cl.y_scaler) if channel else arr

    combined = np.vstack([_flatten(W, real_idx), _flatten(prior, np.arange(n))])
    # perplexity precisa ser menor que o nº de amostras de cada grupo
    perplexity = max(5, min(perplexity, (len(combined) - 1) // 3))
    emb = TSNE(n_components=2, perplexity=perplexity, random_state=seed).fit_transform(combined)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(emb[:n, 0], emb[:n, 1], s=28, alpha=.5, color=COLOR_REAL, label=f"real ({split})")
    ax.scatter(emb[n:, 0], emb[n:, 1], s=28, alpha=.5, color=COLOR_B, label=label)
    scope = f"canal {channel}" if channel else "média dos canais"
    sampling = "janelas sobrepostas" if overlapping else "sem sobreposição"
    ax.set_title(title or f"t-SNE — cliente {cl.cid} ({cl.args.loc}), {split}, {scope}\n"
                          f"n={n} por grupo, {sampling}", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    suffix = channel or "meanchannels"
    out_path = out_path or f'etc/TimeVAE/{cl.args.loc}/results/plots/{cl.cid}_tsne_{split}_{suffix}.png'
    outdir = os.path.dirname(out_path)
    if outdir:
        mkdir_if_not_exists(outdir)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log(INFO, f"{cl.cid}: t-SNE plot saved to {out_path}")
    return out_path


def plot_baseline_vs_opt(cl, gen_base, gen_opt, latent_dim_base: int, latent_dim_opt: int,
                         split: str = "test", **kwargs) -> Optional[str]:
    """Figura 2x2 `<cid>_compare_baseline_vs_opt_<split>.png`: gerador baseline vs o otimizado
    pela HPO (Optuna). É o diagnóstico usado para concluir que a HPO não bate o baseline."""
    title = kwargs.pop("title", f"Cliente {cl.cid} ({cl.args.loc}) — real vs sintético no "
                                f"{split.upper()} | baseline (ld{latent_dim_base}) vs otimizado (ld{latent_dim_opt})")
    out_path = kwargs.pop("out_path",
                          f'etc/TimeVAE/{cl.args.loc}/results/plots/{cl.cid}_compare_baseline_vs_opt_{split}.png')
    return plot_generator_comparison(cl, gen_base, gen_opt, latent_dim_base, latent_dim_opt, split=split,
                                     label_a="baseline", label_b="otimizado", title=title,
                                     out_path=out_path, full=True, **kwargs)


def plot_loss_ablation(cl, gen_original, gen_structural, latent_dim: int, split: str = "test",
                       var_wt: float = 0.0, grad_wt: float = 0.0, **kwargs) -> Optional[str]:
    """Figura 1x2 `<cid>_loss_ablation_<split>.png`: efeito dos termos estruturais da loss de
    reconstrução (`var_wt`/`grad_wt`, ver ClientLearning._get_reconstruction_loss) contra a
    loss original, com o mesmo `latent_dim` nos dois geradores."""
    title = kwargs.pop("title", f"Cliente {cl.cid} — efeito da loss estrutural "
                                f"(ld{latent_dim}, var_wt={var_wt}, grad_wt={grad_wt})")
    out_path = kwargs.pop("out_path",
                          f'etc/TimeVAE/{cl.args.loc}/results/plots/{cl.cid}_loss_ablation_{split}.png')
    return plot_generator_comparison(cl, gen_original, gen_structural, latent_dim, latent_dim, split=split,
                                     label_a="baseline old", label_b="estrutural new", title=title,
                                     out_path=out_path, full=False, **kwargs)
