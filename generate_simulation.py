import yaml
from argparse import ArgumentParser
from logging import INFO
from typing import List, Tuple
from src.utils.logger import log


def _create_compose(location: str, cids: List[int], host_port: Tuple[str, int], clients_per_round: int, max_rounds: int, mem_fraction: float, epochs: int, batch_size: int):
    host, port = host_port
    services = {
        "fl-server": {
            "build": ".",
            "image": "fl-simulation-img",
            "container_name": "fl_server",
            "volumes": [
                "./etc:/app/etc",  # Onde os modelos e CSVs globais são salvos
            ],
            "command": f"python app-server.py --host {host} --required_clients {len(cids)} --clients_per_round {clients_per_round} --max_rounds {max_rounds}",
            "runtime": "nvidia",
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]
                    }
                }
            },
            "ports": [f"{port}:{port}"],
            # "healthcheck": {
            #     "test": ["CMD", "python", "-c", f"import socket; s = socket.socket(); s.connect(({host}, {port}))"],
            #     "interval": "5s",
            #     "timeout": "5s",
            #     "retries": 5,
            #     "start_period": "10s"
            # },
            "environment": [
                "NVIDIA_VISIBLE_DEVICES=all",
                "PYTORCH_ALLOC_CONF=expandable_segments:True",
                "MALLOC_ARENA_MAX=2",
                "OMP_NUM_THREADS=1",
            ],
            "ipc": 'host',
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
                "./etc:/app/etc",  # Onde os modelos e CSVs globais são salvos
            ],
            "depends_on": {
                "fl-server": {"condition": "service_started"}
            },
            "command": f"""python app-client.py --mongo_uri "mongodb://fl-server:27017/" --loc="{location}" --data_path "dataset/pecanstreet/15min/{location}/train/" --test_path "dataset/pecanstreet/15min/{location}/test/"  --host fl-server --filter_bs {c} --epochs {epochs} --batch_size {batch_size} --num_workers 0""",
            "environment": [
                "NVIDIA_VISIBLE_DEVICES=all",
                "CUBLAS_WORKSPACE_CONFIG=:4096:8",
                "PYTORCH_ALLOC_CONF=expandable_segments:True",
                "MALLOC_ARENA_MAX=2",
                "OMP_NUM_THREADS=1",
                f"GPU_FRACTION={mem_fraction}",
                f"CLIENT_ID={c}"
            ],
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]
                    }
                }
            },
            "ipc": 'host',
            "networks": ["fl-network"]
        }

    compose_data = {
        "services": services,
        "networks": {
            "fl-network": {"driver": "bridge"}
        }
    }

    output_file = f"docker-compose.gpu.{location}.yml"
    with open(output_file, "w") as f:
        yaml.dump(compose_data, f, sort_keys=False)

    log(INFO, f"Docker file for {location} created")
    log(INFO, f"Clients available {len(cids)}")
    log(INFO, f"Run docker compose -f {output_file} up --build")



def main():
    parser = ArgumentParser()
    parser.add_argument('--loc', type=str, default='austin', help="[austin, california, newyork, puertorico]")
    parser.add_argument('--host', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=65432)
    parser.add_argument('--clients_per_round', type=int, default=5)
    parser.add_argument('--max_rounds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()

    localities = {
        "austin": [8156, 2335, 9922, 3039, 4031, 8386, 7951, 3538, 9160, 5746, 1642, 6139, 2361, 3456, 9019, 661, 2818, 7800, 7536, 4767, 8565, 4373, 7901, 7719, 9278],
        "california": [8342, 7114, 7062, 6547, 8733, 9612, 9836, 9775, 4495, 9213, 4934, 2606, 8061, 3687, 1731, 3938, 1450, 1524, 3864, 203, 5938, 6377, 8574],
        "newyork": [1417, 142, 3000, 3996, 4550, 2096, 5058, 387, 2318, 5997, 5982, 914, 4283, 3700, 3517, 3488, 27, 5587, 1240, 2358, 558, 5679, 1222, 950, 9053],
        "puertorico": [8235,3835,7667,3987,11386,10887,6056,939,4046,11126,10826,1092,1286,1245,7778,2006,2159,11933,2921,8997,371,9911,4204,4749,6406]
    }
    try:
        cids = localities[args.loc]
        gpu_mem_fraction = .75 / len(cids) # 3% para cada cliente = 75% da GPU
    except:
        raise ValueError(f"Location not exist. Try austin, california, newyork or puertorico.")

    _create_compose(location=args.loc, cids=cids, host_port=(args.host, args.port), clients_per_round=args.clients_per_round,
                    max_rounds=args.max_rounds, mem_fraction=gpu_mem_fraction, epochs=args.epochs,
                    batch_size=args.batch_size)
if __name__ == "__main__":
    main()