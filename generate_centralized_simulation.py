import os
import yaml
from argparse import ArgumentParser
from logging import INFO, WARNING
from src.utils.logger import log

# Import tardio e opcional: get_available_clients_location vive em
# src/utils/functions.py, que importa matplotlib no topo. Gerar um compose nao
# deveria exigir o ambiente de treino inteiro, entao a contagem de execucoes e
# so informativa e some se a dependencia nao estiver disponivel.
try:
    from src.utils.functions import get_available_clients_location
except ImportError:
    get_available_clients_location = None

def _owner(path):
    """Dono de um caminho, em texto, para a mensagem de erro de permissao."""
    try:
        import pwd, grp
        st = os.stat(path)
        return f"{pwd.getpwuid(st.st_uid).pw_name}:{grp.getgrgid(st.st_gid).gr_name}"
    except Exception:
        return "desconhecido"


LOCATIONS = ["austin", "california", "newyork", "puertorico"]
MODELS = ["rnn", "lstm", "gru", "cnn"]


def _runner_command(loc: str, models: str, scope: str, jobs: int, epochs: int, batch_size: int,
                    lr: float, optimizer: str, seed: int, out_root: str, bootstrap: bool,
                    force: bool, no_retry: bool, notify: bool):
    """Monta a linha do run_centralized.sh que o container executa."""
    cmd = ["bash", "run_centralized.sh",
           "-loc", loc,
           "-models", models,
           "-scope", scope,
           "-jobs", str(jobs),
           "-epochs", str(epochs),
           "-batch_size", str(batch_size),
           "-lr", str(lr),
           "-optimizer", optimizer,
           "-seed", str(seed),
           "-out_root", out_root]
    if bootstrap:
        cmd.append("-bootstrap")
    if force:
        cmd.append("-force")
    if no_retry:
        cmd.append("-no_retry")
    if notify:
        cmd.append("-notify")
    return cmd


def _create_compose(loc: str, models: str, scope: str, jobs: int, epochs: int, batch_size: int,
                    lr: float, optimizer: str, seed: int, out_root: str, bootstrap: bool,
                    force: bool, no_retry: bool, notify: bool, output: str = None):
    # Os diretorios montados TEM de existir antes do compose subir. Um bind
    # mount cujo caminho no host nao existe e criado pelo Docker como root:root,
    # e o container (que roda com o uid do usuario) nao consegue escrever nele.
    # Mesma razao do mkdir_if_not_exists("lock_dir") no generate_simulation.py.
    unwritable = []
    for d in ("etc", out_root, "dataset"):
        try:
            os.makedirs(d, exist_ok=True)
        except PermissionError:
            unwritable.append(d)
            continue
        # Criar nao basta: o diretorio pode ja existir de uma execucao anterior
        # que o Docker criou como root. Nesse caso makedirs(exist_ok=True) passa
        # silenciosamente e o erro so aparece la dentro do container.
        if not os.access(d, os.W_OK):
            unwritable.append(d)

    if unwritable:
        st = ", ".join(f"{d} (dono {_owner(d)})" for d in unwritable)
        log(WARNING, f"Sem permissao de escrita em: {st}")
        log(WARNING, f"O container roda como {os.getuid()}:{os.getgid()} e nao vai conseguir "
                     f"escrever ai. Isso acontece quando o Docker criou o volume como root "
                     f"porque o caminho nao existia no host. Corrija com:")
        log(WARNING, f"    sudo chown -R {os.getuid()}:{os.getgid()} {' '.join(unwritable)}")
        log(WARNING, f"    # ou, sem sudo:")
        log(WARNING, f"    docker run --rm -v \"$PWD:/x\" alpine chown -R "
                     f"{os.getuid()}:{os.getgid()} {' '.join('/x/' + d for d in unwritable)}")

    # Opcao A: UM servico rodando o lote inteiro, com o paralelismo controlado
    # pelo -jobs do run_centralized.sh. A GPU e uma so, entao um container por
    # cliente (como no federado) nao compraria paralelismo nenhum - so jogaria
    # fora a retomada, a retentativa de OOM e a trava por job do runner.
    services = {
        "centralized": {
            "build": ".",
            "image": "fl-simulation-img",
            "runtime": "nvidia",
            # Sem container_name: `docker compose run` gera um nome proprio, o
            # que permite mais de uma invocacao sem colisao.
            "volumes": [
                # Recebe os artefatos e os logs por job (<out_root>/_runlogs/...).
                "./etc:/app/etc",
                # ATENCAO: leitura E escrita, ao contrario do cliente federado
                # que monta :ro. O -bootstrap escreve aqui: os community.csv
                # (build_community_dataset.py) e os .npy (migrate_data_numpy.py).
                "./dataset:/app/dataset"
            ],
            "environment": [
                "NVIDIA_VISIBLE_DEVICES=all",
                # seed_all liga torch.use_deterministic_algorithms(True).
                "CUBLAS_WORKSPACE_CONFIG=:4096:8",
                "PYTORCH_ALLOC_CONF=expandable_segments:True",
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                "MALLOC_ARENA_MAX=2",
                "OMP_NUM_THREADS=1",
                # O runner tem como default o python do conda do host.
                "PYTHON=python",
                # Com um user nao-root o HOME do container (/) nao e gravavel,
                # e matplotlib tenta criar /.config/matplotlib no import.
                "HOME=/tmp",
                "MPLCONFIGDIR=/tmp/matplotlib",
                "NOTIFY_WEBHOOK_URL=${NOTIFY_WEBHOOK_URL}",
            ],
            # Escreve em ./etc e ./dataset montados do host: sem isto os
            # arquivos saem como root e o usuario nao mexe neles depois.
            "user": f"{os.getuid()}:{os.getgid()}",
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]
                    }
                }
            },
            "shm_size": '1gb',
            "restart": 'no',
            "command": _runner_command(loc, models, scope, jobs, epochs, batch_size, lr,
                                       optimizer, seed, out_root, bootstrap, force,
                                       no_retry, notify)
        }
    }

    compose_data = {"services": services}

    output_file = output or f"docker-compose.centralized.{loc}.yml"
    with open(output_file, "w") as f:
        yaml.dump(compose_data, f, sort_keys=False, default_flow_style=False)

    log(INFO, f"Compose do treino centralizado criado em {output_file}")
    log(INFO, f"Escopo: {scope} | regioes: {loc} | modelos: {models}")
    if get_available_clients_location is not None:
        locs = LOCATIONS if loc == "all" else [loc]
        n_clients = 0 if scope == "community" else sum(len(get_available_clients_location(l)) for l in locs)
        n_entities = n_clients + (0 if scope == "clients" else len(locs))
        n_models = len(models.split())
        log(INFO, f"Execucoes: {n_entities * n_models} ({n_entities} entidades x "
                  f"{n_models} arquiteturas), {jobs} em paralelo, {epochs} epocas")
    log(INFO, f"Rode: docker compose -f {output_file} up --build")
    log(INFO, f"Acompanhe: bash progress.sh -w -out_root {out_root}")
    return output_file


def main():
    parser = ArgumentParser(
        description="Gera o docker-compose do treino centralizado (clientes individuais "
                    "e/ou serie comunitaria). Equivalente do generate_simulation.py "
                    "para o lado centralizado.")
    parser.add_argument('--loc', type=str, default='all',
                        help="[austin, california, newyork, puertorico, all]")
    parser.add_argument('--models', type=str, default='rnn lstm gru',
                        help='Lista entre aspas, ex.: "rnn lstm gru cnn"')
    parser.add_argument('--scope', type=str, default='both',
                        help="[clients, community, both]")
    parser.add_argument('--jobs', type=int, default=3,
                        help="Execucoes em paralelo. ~2,5 GiB de VRAM por job com batch 1024.")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_root', type=str, default='etc/in')
    parser.add_argument('--bootstrap', action='store_true',
                        help="Gera os community.csv que faltarem e migra os .npy antes de treinar.")
    parser.add_argument('--force', action='store_true',
                        help="Reexecuta jobs cujos artefatos ja existem.")
    parser.add_argument('--no_retry', action='store_true',
                        help="Nao repete em serie os jobs que falharem por falta de VRAM.")
    parser.add_argument('--enable_notifications', action='store_true',
                        help="Envia o sumario final pelo NOTIFY_WEBHOOK_URL. Desligado por padrao.")
    parser.add_argument('--output', type=str, default=None,
                        help="Caminho do arquivo gerado (default: docker-compose.centralized.<loc>.yml)")
    args = parser.parse_args()

    if args.loc != "all" and args.loc not in LOCATIONS:
        raise ValueError(f"Localidade '{args.loc}' nao existe. Use {', '.join(LOCATIONS)} ou all.")
    if args.scope not in ("clients", "community", "both"):
        raise ValueError(f"Escopo '{args.scope}' invalido. Use clients, community ou both.")
    for m in args.models.split():
        if m not in MODELS:
            raise ValueError(f"Modelo '{m}' nao implementado. Use {', '.join(MODELS)}.")
    if args.jobs < 1:
        raise ValueError("--jobs precisa ser >= 1.")

    _create_compose(loc=args.loc, models=args.models, scope=args.scope, jobs=args.jobs,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    optimizer=args.optimizer, seed=args.seed, out_root=args.out_root,
                    bootstrap=args.bootstrap, force=args.force, no_retry=args.no_retry,
                    notify=args.enable_notifications, output=args.output)


if __name__ == "__main__":
    main()
