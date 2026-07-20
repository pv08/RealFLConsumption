import yaml
from argparse import ArgumentParser
from logging import INFO
from typing import List, Tuple
from src.utils.logger import log
from src.utils.functions import mkdir_if_not_exists, get_available_clients_location

def _create_compose(model: str, location: str, cids: List[int], host_port: Tuple[str, int], optimize_clients: bool,  clients_per_round: int, max_rounds: int, gpu_slots: int, epochs: int, batch_size: int, num_workers: int):
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
            "command": f"python app-server.py --host {host} {'--optimize_clients' if optimize_clients else ''} --required_clients {len(cids)} --clients_per_round {clients_per_round} --max_rounds {max_rounds}",
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
                f"WANDB_GROUP={location}-{model}-{'optimized' if optimize_clients else 'not-optimized'}",
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
            "command": f"""python app-client.py --model_name {model} --filter_bs {c} --epochs {epochs} --batch_size {batch_size} --num_workers {num_workers} --loc="{location}" --gpu_slots="{gpu_slots}" --data_path "dataset/pecanstreet/15min/{location}/train/" --test_path "dataset/pecanstreet/15min/{location}/test/"  --host fl-server""",
            "environment": [
                "WANDB_API_KEY=${WANDB_API_KEY}",
                "WANDB_PROJECT=${WANDB_PROJECT}",
                f"WANDB_GROUP={location}-{model}-{'optimized' if optimize_clients else 'not-optimized'}",
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

    output_file = f"docker-compose.gpu.{model}.{location}.yml"
    with open(output_file, "w") as f:
        yaml.dump(compose_data, f, sort_keys=False)

    log(INFO, f"Docker file for {location} created")
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


    parser.add_argument('--gpu_slots', type=int, default=1)

    args = parser.parse_args()
    cids = get_available_clients_location(args.loc)

    _create_compose(model=args.model_name, location=args.loc, cids=cids,
                    host_port=(args.host, args.port), optimize_clients=args.optimize_clients,
                    clients_per_round=args.clients_per_round,
                    max_rounds=args.max_rounds, gpu_slots=args.gpu_slots, epochs=args.epochs,
                    batch_size=args.batch_size, num_workers=args.num_workers)
if __name__ == "__main__":
    main()