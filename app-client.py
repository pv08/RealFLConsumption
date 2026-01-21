import selectors
import socket
import numpy as np
import torch as T
import traceback
import time
from logging import INFO, WARNING, ERROR
from src.models.rnn import RNN
from src.comm import libclient
from src.utils.logger import log
from argparse import ArgumentParser
from src.individual_training import IndividualTraining
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

    # 2. Data args
    parser.add_argument("--data_path", type=str, default='dataset/ufjf-meters/trafos/train/')
    parser.add_argument("--test_path", type=str, default='dataset/ufjf-meters/trafos/test/')
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--targets", type=list, default=['P'])
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default='adamw')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--early_stopping", type=bool, default=False)
    parser.add_argument("--patience", type=int, default=50)

    # 4. Device args
    parser.add_argument("--cuda", type=bool, default=T.cuda.is_available())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=T.device('cuda:0' if T.cuda.is_available() else 'cpu'))

    args = parser.parse_args()

    host, port = args.host, args.port
    socket.setdefaulttimeout(3600)
    # trainer = IndividualTraining(args=args)
    model = RNN(device=args.device, input_dim=1)  # Usando mock para exemplo

    log(INFO, f"Client {args.filter_bs} initiated.")
    # req_checkin = create_request(action="check_in", value=args.filter_bs)
    # message = start_connection(host, port, req_checkin)

    try:
        while True:
            req = create_request("check_in", {"client_id":args.filter_bs, "value": None})
            log(INFO, f"Connecting to {args.host}:{args.port} and waiting selection...")
            resp = send_and_wait(args.host, args.port, req)

            if not resp:
                log(ERROR, "Error in communication. Trying again...")
                continue

            action = resp.get("action")
            data = resp.get("data")

            if action == "stop":
                log(INFO, f"[{args.filter_bs}] Stop signal received from server.")
                log(WARNING, "Finishing simulation in client side.")
                break  # Sai do While True
            elif action == "evaluate":
                log(INFO, f"Evaluating global model at {resp.get('phase', 'N/A')} phase")
                #TODO{Coloca a rotina de avaliação do cliente e retorna as métricas. Vai passar os pesos do modelo global inseridos no pacote}
                req_m = create_request("send_metrics", {"client_id": args.filter_bs, "value": {"mse": 0.001}})
                send_and_wait(host, port, req_m)
            elif action == "train":
                log(INFO, f"Starting training...")

                # A. Carrega Pesos Globais e Treina (Uso intensivo de GPU)
                # revised_weights = trainer.fit(global_weights=data)
                # teste:
                revised_weights=[val.cpu().numpy() for _, val in model.state_dict().items()]
                time.sleep(2)

                # B. Envia Update
                req_u = create_request("send_update", {"client_id": args.filter_bs, "value": revised_weights})
                ack = send_and_wait(args.host, args.port, req_u)
                log(INFO, f"Update sent. Received from server {args.host}:{args.port}: {ack}")
            else:
                log(ERROR, f"Connection failed or invalid response. Trying again in 5s...")
                time.sleep(5)
    except KeyboardInterrupt:
        log(WARNING, "Finishing Client...")
if __name__ == "__main__":
    main()