"""Gera o docker-compose da matriz de experimentos do TimeVAE generativo (clientes x braços x sementes).

Cada serviço é uma execução de `eval_timevae.py` para um cliente, com um `--arm` e uma `--seed`.
Todos os clientes sobem juntos e a GPU é arbitrada pelo `GPULock` (`--gpu_slots`), exatamente como
os clientes da simulação FL — ver `generate_simulation.py`, que este módulo espelha.

    python generate_timevae_experiments.py --loc austin --gpu_slots 3
    docker compose -f docker-compose.timevae.austin.lstm.yml up --build

Sem `--filter_bs` a matriz cobre TODOS os clientes da localidade
(`get_available_clients_location`). As execuções de um mesmo cliente ficam encadeadas em série,
então o nº de contêineres simultâneos é o nº de clientes, não o total de execuções.

Os braços são os da investigação contra o repositório de referência (ver CLAUDE.md):
  A0  comportamento anterior a todas as mudanças (ponto de comparação)
  A1  loss da referência (só o termo de média por feature) + split de validação embaralhado
  A2  A1 + peso por canal na loss
  A3  A2 + tendência/sazonalidade com janelas alinhadas ao dia
  A4  regime da referência (hidden [50,100,200], batch 16)
  A5  A2 + A4 (peso por canal + regime da referência)
  A6  A5 + geração condicional (CVAE: gera os endógenos, condicionado em clima/calendário)
"""
import yaml
from argparse import ArgumentParser
from logging import INFO

from src.utils.functions import get_available_clients_location, mkdir_if_not_exists
from src.utils.logger import log

CHANNEL_WT = "consumption=20,generation=5"
# Canais endógenos do medidor: os únicos que faz sentido GERAR. Os outros 38 (30 de clima, 8 de
# calendário) são exógenos ou determinísticos e entram como condicionamento.
ENDOGENOUS = "consumption,generation,leg1v,leg2v,prev_consumption,consumption_change"

# Os defaults do eval_timevae.py são hoje os do A6 (condicional, peso por canal, arquitetura da
# referência). Como os braços são o REGISTRO do experimento, cada um declara explicitamente os
# quatro eixos abaixo em vez de herdar defaults — herdar faria um braço mudar de significado
# sempre que um default mudasse, sem nada no arquivo indicando isso.
UNCONDITIONAL = ["--gen_channels", "none"]
NO_CHANNEL_WT = ["--channel_wt", "none"]
WITH_CHANNEL_WT = ["--channel_wt", CHANNEL_WT]
BIG_ARCH = ["--hidden_dims", "128", "256", "512", "--batch_size", "256"]   # a nossa, original
REF_ARCH = ["--hidden_dims", "50", "100", "200", "--batch_size", "16"]     # a da referência

ARMS = {
    "A0": {"desc": "baseline: comportamento anterior (dois termos de média, split cronológico)",
           "flags": UNCONDITIONAL + NO_CHANNEL_WT + BIG_ARCH
                    + ["--mean_axis_wt", "1.0", "--no_gen_val_shuffle"]},
    "A1": {"desc": "loss da referência (mean_time_wt=0) + split embaralhado",
           "flags": UNCONDITIONAL + NO_CHANNEL_WT + BIG_ARCH
                    + ["--mean_time_wt", "0.0", "--mean_feat_wt", "1.0"]},
    "A2": {"desc": "A1 + peso por canal (consumo ficava com ~1% da loss)",
           "flags": UNCONDITIONAL + WITH_CHANNEL_WT + BIG_ARCH},
    # O TRTS ajusta o gerador no split de teste, que cobre ~1 dia; com --align_to_day sobra
    # 1 janela e não há como treinar. Por isso este braço é só TSTR.
    "A3": {"desc": "A2 + tendência/sazonalidade com janelas alinhadas ao dia (só TSTR)",
           "flags": UNCONDITIONAL + WITH_CHANNEL_WT + BIG_ARCH
                    + ["--mode", "TSTR", "--trend_poly", "3", "--custom_seas", "24x4", "--align_to_day"]},
    "A4": {"desc": "regime da referência: hidden [50,100,200], batch 16 (bem mais lento)",
           "flags": UNCONDITIONAL + NO_CHANNEL_WT + REF_ARCH},
    "A5": {"desc": "A2 + A4: peso por canal + regime da referência",
           "flags": UNCONDITIONAL + WITH_CHANNEL_WT + REF_ARCH},
    "A6": {"desc": "A5 + geração condicional: modela os 6 canais endógenos, condicionado em "
                   "clima/calendário (que vêm de janelas reais). É o DEFAULT do eval_timevae.py.",
           "flags": WITH_CHANNEL_WT + REF_ARCH + ["--gen_channels", ENDOGENOUS, "--cond_dim", "32"]},
    "BASELINE": {"desc": "Apenas treinamento centralizado local do cliente, gerador não é treinado.",
                 "flags": UNCONDITIONAL + NO_CHANNEL_WT + REF_ARCH},
}


def run_tag(arm, seed, n_seeds, holdout_days=0):
    """Sufixo de semente só quando há réplicas.

    Com uma semente só o nome fica `A5`, igual ao dos braços já rodados antes da matriz de
    sementes existir — assim os CSVs continuam comparáveis lado a lado. A coluna `seed` do CSV
    registra a semente de qualquer forma.

    O sufixo `h` marca as execuções com bloco contíguo reservado: elas treinam em menos dados,
    então não podem cair no mesmo arquivo das execuções sem reserva.
    """
    base = arm if holdout_days <= 0 else f"{arm}h"
    return base if n_seeds == 1 else f"{base}-s{seed}"


def build_service(cid, arm, seed, args, previous, n_seeds):
    tag = run_tag(arm, seed, n_seeds, args.gen_holdout_days)
    service = {
        "build": {"context": ".", "dockerfile": "Dockerfile.timevae"},
        "image": "timevae-eval-img",
        "container_name": f"timevae_{args.model_name}_{args.loc}_{cid}_{arm.lower()}_s{seed}",
        "runtime": "nvidia",
        # ./lock_dir é o MESMO diretório do host em todos os serviços: é isso que faz o flock do
        # GPULock valer entre contêineres (mesmo bind que generate_simulation.py usa).
        "volumes": ["./etc:/app/etc", "./lock_dir:/app/lock_dir", "./dataset:/app/dataset:ro"],
        "environment": [
            "NVIDIA_VISIBLE_DEVICES=all",
            # Redundante com o ENV da imagem, mas explícito: sem isto o backward aborta em CUDA.
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "PYTORCH_ALLOC_CONF=expandable_segments:True",
            "OMP_NUM_THREADS=1",
            "MALLOC_ARENA_MAX=2",
        ],
        "deploy": {"resources": {"reservations": {
            "devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]}}},
        "shm_size": "1gb",
        "restart": "no",
        "entrypoint": [
            "python", "eval_timevae.py",
            "--loc", args.loc,
            "--filter_bs", str(cid),
            "--model_name", args.model_name,
            "--data_path", f"dataset/pecanstreet/15min/{args.loc}/train/",
            "--test_path", f"dataset/pecanstreet/15min/{args.loc}/test/",
            "--mode", args.mode,
            "--timevae_epochs", str(args.timevae_epochs),
            "--r_epochs", str(args.r_epochs),
            "--seed", str(seed),
            "--arm", arm,
            "--run_tag", tag,
            "--gpu_slots", str(args.gpu_slots),
        ] + (["--r_batch_size", str(args.r_batch_size)] if args.r_batch_size else [])
          + (["--gen_holdout_days", str(args.gen_holdout_days)] if args.gen_holdout_days > 0 else [])
          + (["--plots"] if args.plots else []),
        "command": ARMS[arm]["flags"] + (["--batch_size", str(args.batch_size)] if args.batch_size else []),
    }
    if previous:
        # Encadeia apenas DENTRO de um cliente: as execuções (braço x semente) de um mesmo cid
        # rodam em série, e os clientes sobem em paralelo. Sem isso, uma matriz cheia (7 braços x
        # 3 sementes x 25 clientes) tentaria subir 525 contêineres para deixá-los todos parados no
        # flock; a concorrência real quem limita é o --gpu_slots.
        service["depends_on"] = {previous: {"condition": "service_completed_successfully"}}
    return service


def main():
    parser = ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--arms", type=str, nargs="+", default=list(ARMS),
                        help=f"Subconjunto de {list(ARMS)}.")
    parser.add_argument("--loc", type=str, default="austin")
    parser.add_argument("--filter_bs", type=int, nargs="+", default=None,
                        help="Clientes a avaliar. Omitido = todos os da localidade "
                             "(get_available_clients_location).")
    parser.add_argument("--gpu_slots", type=int, default=1,
                        help="Quantas execuções podem treinar na GPU ao mesmo tempo. Vai para o "
                             "--gpu_slots de cada eval_timevae.py, que arbitra via GPULock sobre "
                             "./lock_dir — o mesmo mecanismo dos clientes da simulação FL.")
    parser.add_argument("--model_name", type=str, default="lstm")
    parser.add_argument("--mode", type=str, default="both", help="[both, TSTR, TRTS, baseline]")
    parser.add_argument("--timevae_epochs", type=int, default=200)
    parser.add_argument("--r_epochs", type=int, default=20)
    parser.add_argument("--r_batch_size", type=int, default=None,
                        help="Tamanho do lote do regressor. Se omitido, usa o default de eval_timevae.py.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Tamanho do lote do gerador. Se informado, sobrepõe o default do braço (arm).")
    parser.add_argument("--gen_holdout_days", type=int, default=0,
                        help="Reserva os últimos N dias para avaliar o gerador fora da amostra "
                             "(excluídos do treino + colchão). Os run_tags ganham sufixo 'h'.")
    parser.add_argument("--plots", action="store_true", default=True)
    parser.add_argument("--no_plots", dest="plots", action="store_false",
                        help="Pula os gráficos (mais rápido; o t-SNE é O(n²)).")
    parser.add_argument("--out", type=str, default=None,
                        help="Default: docker-compose.timevae.<loc>.<modelo>.yml")
    args = parser.parse_args()

    unknown = [a for a in args.arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"Braços desconhecidos: {unknown}. Disponíveis: {list(ARMS)}")

    cids = args.filter_bs or get_available_clients_location(args.loc)
    if args.out is None:
        args.out = f"docker-compose.timevae.{args.loc}.{args.model_name}.yml"
    # O GPULock cria o diretório dentro do contêiner, mas o bind precisa existir no host antes do
    # `up`, senão o Docker o cria como root e o processo do contêiner não consegue escrever.
    mkdir_if_not_exists("lock_dir")

    # Ordem: cliente externo, braço, semente — cada cliente vira uma corrente independente, e
    # dentro dela as réplicas de um braço saem juntas (um braço completo já é comparável mesmo se
    # a execução for interrompida no meio).
    services = {}
    for cid in cids:
        previous = None
        for arm in args.arms:
            for seed in args.seeds:
                name = f"tv-{cid}-{arm.lower()}-s{seed}"
                services[name] = build_service(cid, arm, seed, args, previous, len(args.seeds))
                previous = name

    runs_per_client = len(args.arms) * len(args.seeds)
    header = (
        "# GERADO POR generate_timevae_experiments.py — NÃO EDITE À MÃO.\n"
        f"# Localidade: {args.loc} | modelo: {args.model_name} | clientes: {len(cids)} | "
        f"braços: {', '.join(args.arms)} | sementes: {args.seeds}\n"
        f"# {len(services)} execuções = {len(cids)} clientes x {runs_per_client} por cliente.\n"
        "#\n"
        "#   docker compose -f %s up --build\n"
        "#\n"
        f"# Os {len(cids)} clientes sobem em paralelo; as {runs_per_client} execução(ões) de um mesmo\n"
        "# cliente ficam em série (service_completed_successfully). Quem divide a GPU de fato é o\n"
        f"# GPULock: no máximo {args.gpu_slots} execução(ões) treinando por vez (--gpu_slots),\n"
        "# via flock em ./lock_dir. Para uma execução isolada:\n"
        "#   docker compose -f %s up --no-deps <serviço>\n"
        "#\n"
        f"# Saídas por execução, com <TAG> = "
        f"{'<braço>' if len(args.seeds) == 1 else '<braço>-s<semente>'}:\n"
        "#   etc/TimeVAE/<loc>/results/<Model>/<cid>_{TSTR,TRTS}_<TAG>_metrics_ld<d>.csv\n"
        "#   etc/TimeVAE/<loc>/results/plots/<cid>_tsne_*_<TAG>.png\n"
        % (args.out, args.out)
    )
    for arm in args.arms:
        header += f"#   {arm}: {ARMS[arm]['desc']}\n"

    with open(args.out, "w") as f:
        f.write(header)
        yaml.dump({"services": services}, f, sort_keys=False, default_flow_style=False, width=1000)

    log(INFO, f"{len(services)} serviços ({len(cids)} clientes x {len(args.arms)} braços x "
              f"{len(args.seeds)} sementes) escritos em {args.out}")
    log(INFO, f"Concorrência na GPU: {args.gpu_slots} slot(s); {len(cids)} contêineres simultâneos")


if __name__ == "__main__":
    main()
