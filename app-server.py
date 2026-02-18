import selectors
import socket
import traceback
from logging import INFO, ERROR, WARNING
from src.comm import libserver
from src.base.selection_strategy import BaseSelectionStrategy, RandomSelection
from src.base.aggregation_strategy import Aggregator
from src.fl_manager import FLServerState

from src.utils.logger import log
from argparse import ArgumentParser


sel = selectors.DefaultSelector()


def accept_wrapper(sock, fl_state):
    conn, addr = sock.accept()  # Should be ready to read
    log(INFO, f"Accepted connection from {addr}")
    conn.setblocking(False)
    message = libserver.Message(sel, conn, addr, fl_state)
    sel.register(conn, selectors.EVENT_READ, data=message)



def main():
    parser = ArgumentParser()
    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=65432)
    parser.add_argument('--clients_per_round', type=int, default=1)
    parser.add_argument('--optimize_clients', action='store_true')
    parser.add_argument('--required_clients', type=int, default=1)
    parser.add_argument('--max_rounds', type=int, default=2)
    parser.add_argument("--aggregation", type=str, default="fedavg")
    args = parser.parse_args()
    print(args)
    host, port = args.host, args.port
    strategy = RandomSelection()
    aggregation = Aggregator(aggregation_alg=args.aggregation)
    fl_state = FLServerState(selection_strategy=strategy, aggr_strategy=aggregation, required_clients=args.required_clients, clients_per_round=args.clients_per_round, max_rounds=args.max_rounds, optimize_clients=args.optimize_clients)
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
            if fl_state.simulation_over and len(fl_state.tests_received) >=  fl_state.required_clients:
                log(INFO, "Simulação concluída. Encerrando servidor...")
                break
    except KeyboardInterrupt:
        log(WARNING, "Caught keyboard interrupt, exiting")
    finally:
        sel.close()

if __name__ == "__main__":
    main()