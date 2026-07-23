import os
import random
import yaml
from datetime import datetime
from argparse import ArgumentParser
from logging import INFO
from typing import List, Tuple
from src.utils.logger import log
from src.utils.functions import mkdir_if_not_exists, get_available_clients_location

SEED_HISTORY_FILE = "etc/fl/logs/seed_history.log"


def _log_seed_history(location: str, model: str, client_seed: int, server_seed: int,
                      epochs: int, clients_per_round: int, max_rounds: int):
    """Append one row per generated simulation so client (fixed) and server
    (random) seeds stay traceable/reproducible."""
    os.makedirs(os.path.dirname(SEED_HISTORY_FILE), exist_ok=True)
    write_header = not os.path.exists(SEED_HISTORY_FILE)
    with open(SEED_HISTORY_FILE, "a") as f:
        if write_header:
            f.write("datetime,loc,model,client_seed,server_seed,epochs,clients_per_round,max_rounds\n")
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S},{location},{model},"
                f"{client_seed},{server_seed},{epochs},{clients_per_round},{max_rounds}\n")

def _create_compose(model: str, location: str, cids: List[int], host_port: Tuple[str, int], optimize_clients: bool,  clients_per_round: int, max_rounds: int, gpu_slots: int, epochs: int, batch_size: int, num_workers: int, disable_blockchain: bool=False, client_seed: int=0, server_seed: int=0, enable_notifications: bool=False, enable_wandb: bool=False):
    host, port = host_port
    mkdir_if_not_exists("lock_dir")
    services = {
        "fl-server": {
            "build": ".",
            "image": "fl-simulation-img",
            "container_name": "fl_server",
            "volumes": [
                "./etc:/app/etc",
                "./lock_dir:/app/lock_dir",
                "./optuna_data:/app/optuna_db"
            ],
            "command": f"python app-server.py --host {host} {'--optimize_clients' if optimize_clients else ''} {'--disable_blockchain' if disable_blockchain else ''} {'--enable_notifications' if enable_notifications else ''} {'--enable_wandb' if enable_wandb else ''} --required_clients {len(cids)} --clients_per_round {clients_per_round} --max_rounds {max_rounds} --epochs {epochs} --loc=\"{location}\" --seed {server_seed}",
            "runtime": "nvidia",
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]
                    }
                }
            },
            "ports": [f"{port}:{port}"],
            "environment": [
                "WANDB_API_KEY=${WANDB_API_KEY}",
                "WANDB_PROJECT=${WANDB_PROJECT}",
                f"WANDB_GROUP={location}-{model}-{'optimized' if optimize_clients else 'not-optimized'}-cseed{client_seed}-sseed{server_seed}",
                "NOTIFY_WEBHOOK_URL=${NOTIFY_WEBHOOK_URL}",
                "NVIDIA_VISIBLE_DEVICES=all",
                "PYTORCH_ALLOC_CONF=expandable_segments:True",
                "MALLOC_ARENA_MAX=2",
                "OMP_NUM_THREADS=1",

            ],
            "shm_size": '1gb',
            "networks": ["fl-network"]
        }
    }

    for c in cids:
        service_name = f"client_{c}"
        services[service_name] = {
            "image": "fl-simulation-img",
            "runtime": "nvidia",
            "container_name": f"fl_client_{c}",
            "volumes": [
                "./etc:/app/etc",
                "./lock_dir:/app/lock_dir",
                "./dataset:/app/dataset:ro"
            ],
            "depends_on": {
                "fl-server": {"condition": "service_started"},
            },
            "command": f"""python app-client.py --model_name {model} --filter_bs {c} --epochs {epochs} --batch_size {batch_size} --num_workers {num_workers} --loc="{location}" --gpu_slots="{gpu_slots}" --data_path "dataset/pecanstreet/15min/{location}/train/" --test_path "dataset/pecanstreet/15min/{location}/test/"  --host fl-server --seed {client_seed}""",
            "environment": [
                "NVIDIA_VISIBLE_DEVICES=all",
                "CUBLAS_WORKSPACE_CONFIG=:4096:8",
                "PYTORCH_ALLOC_CONF=expandable_segments:True",
                "MALLOC_ARENA_MAX=2",
                "OMP_NUM_THREADS=1",
                f"CLIENT_ID={c}"
            ],
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]
                    }
                }
            },
            "shm_size": '1gb',
            "networks": ["fl-network"]
        }

    compose_data = {
        "services": services,
        "networks": {
            "fl-network": {"driver": "bridge"}
        }
    }

    output_file = f"docker-compose.gpu.{model}.{location}.seed{client_seed}.yml"
    with open(output_file, "w") as f:
        yaml.dump(compose_data, f, sort_keys=False)

    _log_seed_history(location=location, model=model, client_seed=client_seed, server_seed=server_seed,
                      epochs=epochs, clients_per_round=clients_per_round, max_rounds=max_rounds)

    log(INFO, f"Docker file for {location} created")
    log(INFO, f"Client seed (fixed): {client_seed} | Server seed (selection/global init): {server_seed}")
    log(INFO, f"Seed history appended to {SEED_HISTORY_FILE}")
    log(INFO, f"Clients available {len(cids)}")
    log(INFO, f"Run docker compose -f {output_file} up --build")



def main():
    parser = ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=65432)

    parser.add_argument('--loc', type=str, default='austin', help="[austin, california, newyork, puertorico]")
    parser.add_argument('--model_name', type=str, default='rnn', help="[rnn, lstm, gru, cnn]")
    parser.add_argument('--max_rounds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--clients_per_round', type=int, default=5)
    parser.add_argument('--optimize_clients', action='store_true')
    parser.add_argument('--disable_blockchain', action='store_true', help="Skip the Blockchain ledger on the server (no per-update hashing/duplicate-replay check, no ledger file written)")
    parser.add_argument('--enable_notifications', action='store_true', help="Send a webhook notification (NOTIFY_WEBHOOK_URL) when the server-side simulation finishes, is interrupted, or crashes. Disabled by default.")
    parser.add_argument('--enable_wandb', action='store_true', help="Log the FL simulation to Weights & Biases (single server-side run). Requires WANDB_API_KEY in the environment. Disabled by default.")
    parser.add_argument('--seed', type=int, default=0,
                        help="Fixed/predefined seed for the clients (reproducible local training).")
    parser.add_argument('--server_seed', type=int, default=None,
                        help="Seed for the server (client selection + global model init). "
                             "If omitted, a random one is drawn so the server behaves non-deterministically.")


    parser.add_argument('--gpu_slots', type=int, default=1)

    args = parser.parse_args()
    cids = get_available_clients_location(args.loc)

    server_seed = args.server_seed if args.server_seed is not None else random.randint(0, 2**31 - 1)

    _create_compose(model=args.model_name, location=args.loc, cids=cids,
                    host_port=(args.host, args.port), optimize_clients=args.optimize_clients,
                    clients_per_round=args.clients_per_round,
                    max_rounds=args.max_rounds, gpu_slots=args.gpu_slots, epochs=args.epochs,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    disable_blockchain=args.disable_blockchain, client_seed=args.seed,
                    server_seed=server_seed,
                    enable_notifications=args.enable_notifications, enable_wandb=args.enable_wandb)
if __name__ == "__main__":
    main()