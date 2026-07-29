"""Avaliação do TimeVAE como modelo generativo via TSTR / TRTS.

Frameworks:
  - TSTR (Train-on-Synthetic, Test-on-Real): treina o regressor em dados SINTÉTICOS
    (amostrados do prior do gerador ajustado no treino real) e avalia no TESTE REAL.
    Mede se o dado gerado é bom o bastante para treinar um previsor útil.
  - TRTS (Train-on-Real, Test-on-Synthetic): treina o regressor no TREINO REAL e avalia
    em dados SINTÉTICOS (amostrados do prior do gerador ajustado no teste real).
    Mede se o modelo treinado no real performa sobre o dado gerado.

O gerador modela janelas de comprimento num_lags+output_dim sobre os canais reais
(consumo incluso); X e y sintéticos saem por separação temporal da janela — sem o
artifício concat_X_y. Ver src/client_learning.py (build_generative_windows / sample_synthetic).
"""
import time
import pandas as pd
import torch as T
import torch.nn as nn
from argparse import ArgumentParser
from logging import INFO
from torch.utils.data import DataLoader, TensorDataset

from src.client_learning import ClientLearning
from src.data import LocalFileDataset
from src.utils.functions import (get_model, inverse_transform_test, mkdir_if_not_exists,
                                 parse_channel_weights, parse_custom_seas, seed_all)
from src.utils.logger import log


def make_loader(X, y, batch_size, shuffle, num_workers):
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def resolve_mean_wts(args):
    """Resolve os pesos dos dois termos de casamento de média.

    Precedência: `--mean_time_wt`/`--mean_feat_wt` explícitos > `--mean_axis_wt` legado (aplicado
    aos dois) > default da referência (só o termo de feature ativo). O termo de tempo é o que a
    implementação de referência deixa comentado: uma reconstrução achatada no nível certo o
    satisfaz, então ele compete com a fidelidade dos picos."""
    if args.mean_time_wt is None and args.mean_feat_wt is None and args.mean_axis_wt is not None:
        return args.mean_axis_wt, args.mean_axis_wt
    time_wt = 0.0 if args.mean_time_wt is None else args.mean_time_wt
    feat_wt = 1.0 if args.mean_feat_wt is None else args.mean_feat_wt
    return time_wt, feat_wt


def gen_kwargs(args, cl):
    """Hiperparâmetros do gerador repassados a sample_synthetic (tag '-opt' quando otimizado)."""
    mean_time_wt, mean_feat_wt = resolve_mean_wts(args)
    # Na geração condicional a loss cobre só os canais modelados, então o vetor de pesos precisa
    # ser restrito a eles — senão o broadcast bateria contra o feat_dim errado.
    gen_idx, _ = cl.generative_channel_split()
    return dict(lr=args.gen_lr, beta=args.beta, kl_anneal_frac=args.kl_anneal_frac,
                reconstruction_wt=args.reconstruction_wt,
                mean_time_wt=mean_time_wt, mean_feat_wt=mean_feat_wt,
                var_wt=args.var_wt, grad_wt=args.grad_wt,
                channel_wt=cl.build_channel_wt(parse_channel_weights(args.channel_wt), channels=gen_idx),
                log_every=args.log_every, tag=getattr(args, "_gen_tag", ""))


def train_regressor(cl, train_loader, val_loader, args):
    """Treina um regressor novo (mesma arquitetura da simulação FL) e devolve o best_model."""
    model = get_model(device=args.device, model=args.model_name, input_dim=cl.input_dim,
                      out_dim=cl.output_dim, lags=args.num_lags)
    best_model, _, _ = cl.train(train_loader=train_loader, val_loader=val_loader, model=model,
                                epochs=args.r_epochs, optimizer=args.r_optimizer, lr=args.r_lr,
                                criterion=args.criterion, early_stopping=False, patience=args.patience,
                                device=args.device)
    return best_model


def evaluate_model(cl, model, loader):
    """Avalia o modelo no loader; devolve métricas escaladas e invertidas (via y_scaler)."""
    loss, mse, rmse, mae, mape, r2, nrmse, pinball, y_true, y_pred = cl.test(
        model, loader, nn.MSELoss(), device=cl.args.device)
    scaled = {"loss": float(loss), "MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae),
              "MAPE": float(mape), "R2": float(r2), "NRMSE": float(nrmse), "pinball": float(pinball)}
    y_true_inv, y_pred_inv = inverse_transform_test(y_true.numpy(), y_pred.numpy(),
                                                    cl.y_scaler, round_preds=False, dims=[0])
    imse, irmse, imae, imape, ir2, inrmse, ipinball = cl.accumulate_metrics(y_true_inv, y_pred_inv)
    inverted = {"MSE": float(imse), "RMSE": float(irmse), "MAE": float(imae), "MAPE": float(imape),
                "R2": float(ir2), "NRMSE": float(inrmse), "pinball": float(ipinball)}
    return scaled, inverted


def save_results(args, mode, model_cls, scaled, inverted):
    outdir = f'etc/TimeVAE/{args.loc}/results/{model_cls}'
    mkdir_if_not_exists(outdir)
    # `run_tag` separa braços de experimento: sem ele todos escreveriam o mesmo arquivo e o
    # último a rodar apagaria os anteriores.
    suffix = f'_{args.run_tag}' if args.run_tag else ''
    row = {"client": args.filter_bs, "mode": mode, "run_tag": args.run_tag, "arm": args.arm,
           "seed": args.seed, "model": model_cls,
           "latent_dim": args.latent_dim, "n_synthetic": args.n_synthetic,
           "timevae_epochs": args.timevae_epochs, "r_epochs": args.r_epochs,
           "channel_wt": args.channel_wt, "mean_time_wt": args.mean_time_wt, "mean_feat_wt": args.mean_feat_wt,
           "mean_axis_wt": args.mean_axis_wt, "trend_poly": args.trend_poly, "custom_seas": args.custom_seas,
           "align_to_day": args.align_to_day, "window_stride": args.window_stride,
           "hidden_dims": str(args.hidden_dims), "batch_size": args.batch_size,
           "gen_val_shuffle": args.gen_val_shuffle,
           "gen_channels": ",".join(args.gen_channels) if args.gen_channels else None,
           "cond_dim": args.cond_dim if args.gen_channels else None}
    row.update({f"scaled_{k}": v for k, v in scaled.items()})
    row.update({f"inv_{k}": v for k, v in inverted.items()})
    path = f'{outdir}/{args.filter_bs}_{mode}{suffix}_metrics_ld{args.latent_dim}.csv'
    pd.DataFrame([row]).to_csv(path, index=False)
    log(INFO, f"[{mode}] metrics saved to {path}")
    log(INFO, f"[{mode}] scaled={scaled}")
    log(INFO, f"[{mode}] inverted={inverted}")


def run_tstr(cl, args):
    log(INFO, "===== TSTR: train synthetic, test real =====")
    X_syn, y_syn = cl.sample_synthetic(n=args.n_synthetic, latent_dim=args.latent_dim,
                                       epochs=args.timevae_epochs, split="train", **gen_kwargs(args, cl))
    n = len(X_syn)
    n_val = max(1, int(0.2 * n))
    perm = T.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_loader = make_loader(X_syn[tr_idx], y_syn[tr_idx], args.r_batch_size, True, args.num_workers)
    val_loader = make_loader(X_syn[val_idx], y_syn[val_idx], args.r_batch_size, False, args.num_workers)

    model = train_regressor(cl, tr_loader, val_loader, args)

    test_ds = LocalFileDataset(client_id=args.filter_bs, _type="test", data_path=args.test_path)
    test_loader = DataLoader(test_ds, batch_size=args.r_batch_size, shuffle=False, num_workers=args.num_workers)
    scaled, inverted = evaluate_model(cl, model, test_loader)
    save_results(args, "TSTR", type(model).__name__, scaled, inverted)


def run_trts(cl, args):
    log(INFO, "===== TRTS: train real, test synthetic =====")
    real_tr = DataLoader(cl.train_dataset, batch_size=args.r_batch_size, shuffle=True, num_workers=args.num_workers)
    real_val = DataLoader(cl.val_dataset, batch_size=args.r_batch_size, shuffle=False, num_workers=args.num_workers)
    model = train_regressor(cl, real_tr, real_val, args)

    X_syn, y_syn = cl.sample_synthetic(n=args.n_synthetic, latent_dim=args.latent_dim,
                                       epochs=args.timevae_epochs, split="test", **gen_kwargs(args, cl))
    syn_loader = make_loader(X_syn, y_syn, args.r_batch_size, False, args.num_workers)
    scaled, inverted = evaluate_model(cl, model, syn_loader)
    save_results(args, "TRTS", type(model).__name__, scaled, inverted)


def run_baseline(cl, args):
    log(INFO, "===== BASELINE: train real, test real =====")
    real_tr = DataLoader(cl.train_dataset, batch_size=args.r_batch_size, shuffle=True, num_workers=args.num_workers)
    real_val = DataLoader(cl.val_dataset, batch_size=args.r_batch_size, shuffle=False, num_workers=args.num_workers)
    model = train_regressor(cl, real_tr, real_val, args)
    test_ds = LocalFileDataset(client_id=args.filter_bs, _type="test", data_path=args.test_path)
    test_loader = DataLoader(test_ds, batch_size=args.r_batch_size, shuffle=False, num_workers=args.num_workers)
    scaled, inverted = evaluate_model(cl, model, test_loader)
    save_results(args, "BASELINE", type(model).__name__, scaled, inverted)


def run_plots(cl, args, splits):
    """Gera os gráficos de diagnóstico do gerador para cada split avaliado.

    O gerador vem do cache (a avaliação acabou de treiná-lo com os mesmos hiperparâmetros, logo
    o mesmo hash de configuração). Para cada split: t-SNE no canal do consumo e t-SNE na média
    dos canais (a forma da referência), mais os painéis de curva/distribuição/perfil/janelas."""
    from src.utils.graphs import plot_generator_comparison, plot_tsne

    tag = getattr(args, "_gen_tag", "")
    label = args.run_tag or ("otimizado" if tag == "-opt" else "gerador")
    suffix = f'_{args.run_tag}' if args.run_tag else tag
    plots_dir = f'etc/TimeVAE/{args.loc}/results/plots'
    for split in splits:
        gen = cl.train_generative_timevae(latent_dim=args.latent_dim, epochs=args.timevae_epochs,
                                          split=split, **gen_kwargs(args, cl))
        for channel in ("consumption", None):
            name = channel or "meanchannels"
            plot_tsne(cl, gen, args.latent_dim, split=split, label=f"{label} (prior)",
                      channel=channel, max_samples=args.tsne_max_samples, seed=args.seed,
                      out_path=f'{plots_dir}/{cl.cid}_tsne_{split}_{name}{suffix}.png')
        # Painéis A-D comparando o gerador consigo mesmo não faz sentido; aqui o "A" é a
        # reconstrução e o "B" o prior do MESMO gerador, então passamos ele nos dois lugares.
        plot_generator_comparison(
            cl, gen, gen, args.latent_dim, args.latent_dim, split=split,
            label_a=label, label_b=label, seed=args.seed,
            title=f"Cliente {cl.cid} ({args.loc}) — {split.upper()} | {label} (ld{args.latent_dim})",
            out_path=f'{plots_dir}/{cl.cid}_generator_{split}{suffix}.png')


def build_parser():
    """O parser do CLI, separado de `main()` para poder ser reusado.

    Scripts de diagnóstico precisam reconstruir exatamente o mesmo `args` de uma execução (o hash
    de configuração do checkpoint depende dele) sem rodar a avaliação inteira."""
    parser = ArgumentParser()
    # Data / client
    parser.add_argument("--loc", type=str, default="austin", help="[austin, california, newyork, puertorico]")
    parser.add_argument("--filter_bs", default=661)
    parser.add_argument("--data_path", type=str, default="dataset/pecanstreet/15min/austin/train/")
    parser.add_argument("--test_path", type=str, default="dataset/pecanstreet/15min/austin/test/")
    parser.add_argument("--num_lags", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="both", help="[TSTR, TRTS, both, baseline]")
    parser.add_argument("--log_every", type=int, default=None,
                        help="Intervalo (em épocas) das linhas de andamento do treino do TimeVAE. "
                             "Default: ~25 linhas por treino. A barra tqdm só aparece em terminal "
                             "interativo; fora dele (docker logs) valem estas linhas.")
    parser.add_argument("--run_tag", type=str, default="",
                        help="Identificador da execução (ex.: A1-s0). Entra no nome dos CSVs e dos "
                             "gráficos; sem ele os braços sobrescrevem uns aos outros.")
    parser.add_argument("--arm", type=str, default="",
                        help="Nome do braço sem a semente (ex.: A1). Vira coluna no CSV, para agrupar "
                             "as réplicas na hora de calcular média e desvio.")

    # Regressor
    parser.add_argument("--model_name", type=str, default="lstm", help="[rnn, lstm, gru, cnn]")
    parser.add_argument("--criterion", type=str, default="mse")
    parser.add_argument("--r_epochs", type=int, default=10)
    parser.add_argument("--r_lr", type=float, default=1e-3)
    parser.add_argument("--r_optimizer", type=str, default="adamw")
    parser.add_argument("--r_batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=50)

    # Generative TimeVAE
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--timevae_epochs", type=int, default=50)
    parser.add_argument("--n_synthetic", type=int, default=None,
                        help="Nº de janelas sintéticas amostradas do prior (default = nº de janelas reais do split).")
    # Os defaults abaixo são os do braço A6, o melhor medido no bloco contíguo reservado
    # (ver CLAUDE.md). Para o comportamento anterior: --hidden_dims 128 256 512 --batch_size 256
    # --channel_wt none --gen_channels none.
    parser.add_argument("--batch_size", type=int, default=16, help="Batch do treino do gerador (VAE).")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[50, 100, 200],
                        help="Filtros do encoder/decoder. O default da referência é 50 100 200.")
    parser.add_argument("--trend_poly", type=int, default=0,
                        help="Grau do polinômio de tendência do decoder (0 = desligado).")
    parser.add_argument("--reconstruction_wt", type=float, default=3.0)
    parser.add_argument("--use_residual_conn", action="store_true", default=True)
    parser.add_argument("--custom_seas", type=str, default=None,
                        help="Sazonalidade do decoder, 'NxL' separados por vírgula (ex.: '24x4' = "
                             "24 estações de 4 passos). Só faz sentido com --align_to_day.")
    # Janelas generativas
    parser.add_argument("--window_stride", type=int, default=1,
                        help="Passo entre janelas generativas: em passos de 15 min normalmente, "
                             "ou em DIAS quando --align_to_day. 1 = todas.")
    parser.add_argument("--align_to_day", action="store_true",
                        help="Mantém só as janelas que começam em 00:00. Necessário para a "
                             "sazonalidade (--custom_seas) ser interpretável; reduz ~96x o nº de janelas.")
    parser.add_argument("--no_gen_val_shuffle", dest="gen_val_shuffle", action="store_false",
                        help="Usa a cauda cronológica como validação do gerador em vez de embaralhar "
                             "(a referência embaralha; embaralhar é o default).")
    parser.set_defaults(gen_val_shuffle=True)
    # Geração condicional
    parser.add_argument("--gen_channels", type=str,
                        default="consumption,generation,leg1v,leg2v,prev_consumption,consumption_change",
                        help="Nomes de canais (separados por vírgula) que o gerador MODELA; os demais "
                             "(clima, calendário) passam a CONDICIONAR o decoder e são copiados de uma "
                             "janela real na amostragem. Precisa incluir 'consumption'. O default são os "
                             "6 canais endógenos do medidor — os únicos que faz sentido gerar. Use "
                             "'none' para um gerador incondicional sobre os 44 canais.")
    parser.add_argument("--cond_dim", type=int, default=32,
                        help="Dimensão do embedding do contexto exógeno (só com --gen_channels).")
    parser.add_argument("--gen_holdout_days", type=int, default=0,
                        help="Reserva os últimos N dias da série (train+val) para avaliar o gerador "
                             "fora da amostra: são excluídos do treino, junto de um colchão de "
                             "seq_len-1 janelas para que nenhuma janela de treino os invada. 0 = "
                             "desligado (comportamento dos braços A0-A6).")
    # Hiperparâmetros de treino do gerador (usados quando NÃO se otimiza; a HPO os sobrepõe)
    parser.add_argument("--gen_lr", type=float, default=1e-3, help="lr do AdamW no treino do gerador.")
    parser.add_argument("--beta", type=float, default=1.0, help="Peso final do KL (β-VAE).")
    parser.add_argument("--kl_anneal_frac", type=float, default=0.0,
                        help="Fração das épocas para subir o KL de 0 a beta (annealing). 0 = desligado.")
    # Forma da loss de reconstrução (melhoria estrutural contra a suavização/perda de picos)
    parser.add_argument("--mean_axis_wt", type=float, default=None,
                        help="Legado: peso único aplicado aos DOIS termos de média. Se omitido, "
                             "valem os defaults da referência (mean_time_wt=0, mean_feat_wt=1).")
    parser.add_argument("--mean_time_wt", type=float, default=None,
                        help="Casamento da média sobre o TEMPO por feature. A referência deixa este "
                             "termo desligado: uma reconstrução achatada no nível certo o satisfaz.")
    parser.add_argument("--mean_feat_wt", type=float, default=None,
                        help="Casamento da média sobre as FEATURES a cada instante (ativo na referência).")
    parser.add_argument("--var_wt", type=float, default=0.0, help="Peso do casamento de VARIÂNCIA temporal.")
    parser.add_argument("--grad_wt", type=float, default=0.0, help="Peso do casamento do GRADIENTE temporal (picos).")
    parser.add_argument("--channel_wt", type=str, default="consumption=20,generation=5",
                        help="Pesos por canal na loss, 'nome=peso' separados por vírgula; os demais "
                             "ficam em 1.0. Com a soma uniforme sobre os 44 canais o consumo fica com "
                             "~1%% da loss, daí o default. Use 'none' para desligar.")

    # Otimização de hiperparâmetros (Optuna) do gerador
    parser.add_argument("--optimize", action="store_true",
                        help="Roda HPO (Optuna) do gerador no split de treino antes de avaliar.")
    parser.add_argument("--n_trials", type=int, default=15)
    parser.add_argument("--hpo_epochs", type=int, default=40, help="Épocas do VAE por trial da HPO.")
    parser.add_argument("--hpo_r_epochs", type=int, default=8, help="Épocas do regressor por trial (objetivo TSTR).")
    parser.add_argument("--hpo_n_synthetic", type=int, default=5000, help="Amostras sintéticas por trial da HPO.")

    # Diagnóstico
    parser.add_argument("--plots", action="store_true",
                        help="Gera t-SNE (real vs prior) e os painéis do gerador em "
                             "etc/TimeVAE/<loc>/results/plots/ ao fim da avaliação.")
    parser.add_argument("--tsne_max_samples", type=int, default=1000,
                        help="Nº máximo de janelas por grupo no t-SNE (custo é O(n²)).")
    parser.add_argument("--allow_legacy_ckpt", action="store_true",
                        help="Permite carregar checkpoints com o nome antigo (<cid>-gen-<split>-latent_dim_<d>.pth), "
                             "de antes do hash de configuração. Só para reproduzir a baseline histórica: a config "
                             "de treino deles não é verificável e NÃO corresponde às flags passadas.")
    return parser


def _disabled(value):
    """'' e 'none' significam AUSENTE, para que se possa desligar um default pela linha de comando.

    Necessário desde que a geração condicional e o peso por canal viraram default: um braço de
    experimento que queira o comportamento antigo precisa dizer isso explicitamente."""
    return value is None or str(value).strip().lower() in ("", "none")


def parse_args(argv=None):
    """Parseia e aplica as derivações que `main()` faz antes de tocar em qualquer modelo."""
    args = build_parser().parse_args(argv)
    args.device = "cuda:0" if T.cuda.is_available() else "cpu"
    # `ClientLearning` (e o TimeVAE de seleção, via app-client.py) lê o atributo `custom_seats`;
    # aqui ele vem parseado da string 'NxL' de `--custom_seas`.
    args.custom_seats = parse_custom_seas(args.custom_seas)
    args.channel_wt = None if _disabled(args.channel_wt) else args.channel_wt
    args.gen_channels = (None if _disabled(args.gen_channels)
                         else [c.strip() for c in args.gen_channels.split(",") if c.strip()])
    return args


def main():
    args = parse_args()
    seed_all(args.seed)
    log(INFO, f"eval_timevae args: {args}")

    start = time.time()
    cl = ClientLearning(args=args, cid=args.filter_bs, seed=args.seed)

    if args.optimize:
        log(INFO, "===== Optuna HPO do gerador (split=train) =====")
        best, best_val = cl.optimize_generative_timevae(
            split="train", n_trials=args.n_trials, hpo_epochs=args.hpo_epochs,
            hpo_r_epochs=args.hpo_r_epochs, hpo_n_synthetic=args.hpo_n_synthetic,
            model_name=args.model_name, full_epochs=args.timevae_epochs, seed=args.seed)
        # Aplica os melhores hiperparâmetros e usa o gerador otimizado ('-opt') na avaliação.
        args.latent_dim = best["latent_dim"]
        args.gen_lr = best["lr"]
        args.beta = best["beta"]
        args.kl_anneal_frac = best["kl_anneal_frac"]
        args.reconstruction_wt = best["reconstruction_wt"]
        args.mean_axis_wt = None  # os pesos separados abaixo têm precedência
        args.mean_time_wt = best["mean_time_wt"]
        args.mean_feat_wt = best["mean_feat_wt"]
        args.var_wt = best["var_wt"]
        args.grad_wt = best["grad_wt"]
        args.channel_wt = f"consumption={best['cons_wt']}"
        args._gen_tag = "-opt"
        log(INFO, f"HPO best (real-val MSE {best_val:.6f}): {best}")

    splits = []
    if args.mode in ("TSTR", "both"):
        run_tstr(cl, args)
        splits.append("train")
    if args.mode in ("TRTS", "both"):
        run_trts(cl, args)
        splits.append("test")
    if args.mode == "baseline":
        run_baseline(cl, args)

    if args.plots and splits:
        log(INFO, "===== Gráficos de diagnóstico =====")
        run_plots(cl, args, splits)

    log(INFO, f"eval_timevae finished in {(time.time() - start) / 60:.2f} min")


if __name__ == "__main__":
    main()
