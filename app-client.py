import selectors
import socket
import os
import torch as T
import time
import copy
from logging import INFO, WARNING, ERROR
from src.comm import libclient
from src.utils.logger import log
from src.utils.gpu_lock import GPULock
from src.utils.process_executor import ProcessExecutor
from argparse import ArgumentParser
from src.client_learning import ClientLearning


sel = selectors.DefaultSelector()


def create_request(action, value):
    return dict(
        type="binary/pickle",
        encoding="binary",
        content=dict(action=action, content=value),
    )


def send_and_wait(host, port, request):
    """Função auxiliar para enviar request e esperar resposta de forma síncrona"""
    message = start_connection(host, port, request)
    response = None

    # Loop local de eventos até ter resposta
    while True:
        events = sel.select(timeout=False)
        if not events:
            # Timeout ou processamento
            continue

        for key, mask in events:
            msg = key.data
            try:
                msg.process_events(mask)
            except Exception:
                msg.close()
                return None

        # Se a mensagem fechou ou terminou, pegamos a resposta
        if not sel.get_map():
            response = message.response
            break

    return response


def start_connection(host, port, request):
    addr = (host, port)
    log(INFO, f"Starting connection to {addr}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    message = libclient.Message(sel, sock, addr, request)
    sel.register(sock, events, data=message)
    return message

def main():
    parser = ArgumentParser()
    # 1. Communication args
    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=65432)
    parser.add_argument('--mongo_uri', type=str, default="mongodb://192.168.1.28:27017/")

    # 2. Data args
    parser.add_argument("--data_path", type=str, default='dataset/pecanstreet/15min/austin/train/')
    parser.add_argument("--test_path", type=str, default='dataset/pecanstreet/15min/austin/test/')
    parser.add_argument("--loc", type=str, default="austin", help="[austin, california, newyork, puertorico]")
    parser.add_argument("--targets", type=list, default=['consumption'])
    parser.add_argument("--num_lags", type=int, default=96)
    parser.add_argument("--filter_bs", default=0)
    parser.add_argument("--identifier", type=str, default='cid')
    parser.add_argument("--nan_constant", type=int, default=0)
    parser.add_argument("--x_scaler", type=str, default='minmax')
    parser.add_argument("--y_scaler", type=str, default='minmax')
    parser.add_argument("--outlier_detection", type=any, default=None)

    # 3. Model args
    parser.add_argument("--criterion", type=str, default='mse')
    parser.add_argument("--model_name", type=str, default='lstm', help='["mlp", "rnn" ,"lstm", "gru", "cnn", "da_encoder_decoder"]')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default='adamw')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--early_stopping", type=bool, default=False)
    parser.add_argument("--patience", type=int, default=50)

    # 4. Device args
    parser.add_argument("--cuda", type=bool, default=T.cuda.is_available())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=T.device('cuda:0' if T.cuda.is_available() else 'cpu'))

    args = parser.parse_args()


    host, port = args.host, args.port
    socket.setdefaulttimeout(3600)
    trainer = ClientLearning(args=args, cid=args.filter_bs, seed=args.seed)
    log(INFO, f"Client {args.filter_bs} initiated -> {args}.")

    try:
        while True:
            req = create_request("check_in", {"client_id":args.filter_bs,
                                              "value": {
                                                  "device": args.device,
                                                  "model_name": args.model_name,
                                                  "input_dim": trainer.input_dim,
                                                  "output_dim": trainer.output_dim,
                                                  "lags": args.num_lags
                                              }
                                              }
                                 )
            log(INFO, f"Connecting to {args.host}:{args.port} and waiting selection...")
            resp = send_and_wait(args.host, args.port, req)

            if not resp:
                log(ERROR, "Error in communication. Trying again...")
                continue

            action = resp.get("action")
            data = resp.get("data")

            if action == "stop":
                log(INFO, f"[{args.filter_bs}] Stop signal received from server. Testing global model on test set.")
                global_model_params = data["weights"]
                results, inverted_values = trainer.test_model(global_model_params)
                req_t = create_request("send_test", {"client_id": args.filter_bs,
                                                        "value": {"results": results, "inverted_results": inverted_values}})
                send_and_wait(host, port, req_t)

                log(WARNING, "Finishing simulation in client side.")
                break

            elif action == "evaluate":
                log(INFO, f"Evaluating global model at {resp.get('phase', 'N/A')} phase")
                global_model_params = data["weights"]
                with GPULock(client_id=args.filter_bs):
                    num_test_instances, test_loss, test_eval_metrics = ProcessExecutor.run_evaluate(
                        args=args,
                        params=global_model_params
                    )

                req_m = create_request("send_metrics", {"client_id": args.filter_bs, "value": {"instances": num_test_instances, "loss": test_loss,
                                                                                                 "metrics": test_eval_metrics}})
                send_and_wait(host, port, req_m)

            elif action == "train":
                start_time = time.time()
                log(INFO, f"Starting training...")
                global_model_params = data["weights"]
                with GPULock(client_id=args.filter_bs):
                    res = ProcessExecutor.run_train(
                        args=args,
                        params=global_model_params
                    )
                    end_time = time.time()
                    training_time = end_time - start_time
                    log(INFO,f"Time spent to train client {args.filter_bs} {training_time} seconds --> {(training_time) / 3600} hours")
                    res += (training_time, )

                    trainer.clean_up()
                # B. Envia Update
                req_u = create_request("send_update", {"client_id": args.filter_bs, "value": res})
                ack = send_and_wait(args.host, args.port, req_u)
                log(INFO, f"Update sent. Received from server {args.host}:{args.port}: {ack}")
            else:
                log(ERROR, f"Connection failed or invalid response. Trying again in 5s...")
                time.sleep(5)
    except KeyboardInterrupt:
        log(WARNING, "Finishing Client...")
if __name__ == "__main__":
    main()