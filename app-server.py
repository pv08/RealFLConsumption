import selectors
import socket
import traceback
import os
from typing import Optional
from logging import INFO, ERROR, WARNING
from src.comm import libserver
from src.base.selection_strategy import RandomSelection, TimeVAE, TimeVAEWeeklyRepresentativeSelection
from src.base.aggregation_strategy import Aggregator
from src.fl_manager import FLServerState
from src.utils.logger import log
from src.utils.functions import seed_all
from argparse import ArgumentParser


sel = selectors.DefaultSelector()


def accept_wrapper(sock, fl_state):
    conn, addr = sock.accept()  # Should be ready to read
    log(INFO, f"Accepted connection from {addr}")
    conn.setblocking(False)
    message = libserver.Message(sel, conn, addr, fl_state)
    sel.register(conn, selectors.EVENT_READ, data=message)

def get_select_strategy(strategy: str="random", cluster_size: Optional[int]=None, seed: int=0):
    if strategy == "random":
        return RandomSelection(seed=seed)
    elif strategy == "fixed-representativeness" and cluster_size is not None:
        return TimeVAE(min_cluster_size=cluster_size)
    elif strategy == "weekly-representativeness" and cluster_size is not None:
        return TimeVAEWeeklyRepresentativeSelection(min_cluster_size=cluster_size)
    else:
        raise KeyError(f"Please, select a client selection strategy valid. {strategy} do not exists. Valid options: ['random', 'fixed-representativeness', 'weekly-representativeness']")


def main():
    parser = ArgumentParser()
    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=65432)
    parser.add_argument('--client_strategy', type=str, default='random', help="['random', 'fixed-representativeness', 'weekly-representativeness']")
    parser.add_argument('--min_cluster_size', type=int, default=2)
    parser.add_argument('--clients_per_round', type=int, default=5)
    parser.add_argument('--optimize_clients', action='store_true')
    parser.add_argument('--disable_blockchain', action='store_true', help="Skip the Blockchain ledger (no per-update hashing/duplicate-replay check, no ledger file written)")
    parser.add_argument('--required_clients', type=int, default=5)
    parser.add_argument('--max_rounds', type=int, default=2)
    parser.add_argument("--aggregation", type=str, default="fedavg")
    parser.add_argument('--wandb_project', type=str, default=os.getenv('WANDB_PROJECT', 'fl_default'))
    parser.add_argument('--wandb_group', type=str, default=os.getenv('WANDB_GROUP', 'default_group'))
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    print(args)
    host, port = args.host, args.port
    strategy = get_select_strategy(strategy=args.client_strategy, cluster_size=args.min_cluster_size, seed=args.seed)
    aggregation = Aggregator(aggregation_alg=args.aggregation)
    wandb_config = {
        'project': args.wandb_project,
        'group': args.wandb_group
    }
    fl_state = FLServerState(selection_strategy=strategy, aggr_strategy=aggregation,
                             required_clients=args.required_clients,
                             clients_per_round=args.clients_per_round,
                             max_rounds=args.max_rounds, optimize_clients=args.optimize_clients,
                             wandb_config=wandb_config, seed=args.seed,
                             disable_blockchain=args.disable_blockchain)
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind((host, port))
    lsock.listen()
    log(INFO, f"Listening on {(host, port)}")
    log(INFO, f"{args.required_clients} clients necessary to train. The selection in based on {args.clients_per_round} clients per round")
    lsock.setblocking(False)
    sel.register(lsock, selectors.EVENT_READ, data=None)

    try:
        while True:
            events = sel.select(timeout=None)
            for key, mask in events:
                if key.data is None:
                    accept_wrapper(key.fileobj, fl_state)
                else:
                    message = key.data
                    try:
                        message.process_events(mask)
                    except RuntimeError as e:
                        if "Peer closed" in str(e):
                            log(INFO, f"Client {message.addr} finished the connection (Check-in/Heartbeat concluded)")
                            message.close()
                        else:
                            log(ERROR, f"RuntimeError com {message.addr}: {e}")
                            message.close()
                    except Exception:
                        log(ERROR,
                            f"Main: Error: Exception for {message.addr}:\n"
                            f"{traceback.format_exc()}"
                        )
                        message.close()
            if fl_state.simulation_over and len(fl_state.tests_received) >= fl_state.required_clients:
                log(INFO, "Simulation concluded. Finishing the server...")
                break
    except KeyboardInterrupt:
        log(WARNING, "Caught keyboard interrupt, exiting")
    finally:
        sel.close()

if __name__ == "__main__":
    main()