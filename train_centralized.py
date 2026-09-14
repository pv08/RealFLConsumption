"""
train_centralized.py

Treinamento centralizado de uma entidade: um cliente individual (`--filter_bs 661`)
ou a serie comunitaria da regiao (`--filter_bs community`). Uma execucao = um par
(entidade, arquitetura). Para percorrer varias, use run_centralized.sh.

Entrypoint novo, que substitui o main_in.py/IndividualTraining apagados em
35b77a2. Em vez de ressuscitar aquele codigo, reaproveita o caminho que ja treina
os clientes federados (`src.client_learning.ClientLearning`): mesmo
pre-processamento (`Processing.make_preprocessing`, via migrate_data_numpy.py),
mesma regra de melhor checkpoint (menor perda de validacao dentro das epocas),
mesmas funcoes de metrica. Assim os dois lados da comparacao federado x
centralizado passam a sair do mesmo codigo, e o centralizado por cliente sai do
mesmo codigo que o comunitario.

Hiperparametros: 200 epocas, AdamW, lr 1e-3, MSE, num_lags 96, minmax nos dois
escaladores, nan_constant 0, early_stopping False, seed 0. O batch_size default e
1024 (o mesmo que generate_simulation.py passa aos clientes federados); os
resultados comunitarios ja entregues usaram 512, que run_community.sh preserva.

Pre-requisitos (por entidade):
    # so para a comunidade:
    python tools/build_community_dataset.py --loc all
    # para qualquer entidade, inclusive a comunidade:
    python migrate_data_numpy.py --loc <loc> --filter_bs <cid|community> \
        --data_path dataset/pecanstreet/15min/<loc>/train/ \
        --test_path dataset/pecanstreet/15min/<loc>/test/

Uso:
    python train_centralized.py --loc austin --model_name lstm --filter_bs 661
    python train_centralized.py --loc austin --model_name lstm --filter_bs community
"""
import os

# seed_all liga torch.use_deterministic_algorithms(True); em CUDA isso exige a
# variavel abaixo, e ela precisa estar no ambiente antes do cuBLAS inicializar.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import time
from argparse import ArgumentParser
from logging import INFO

import h5py
import numpy as np
import torch as T
from sklearn.metrics import mean_squared_error

from src.client_learning import ClientLearning
from src.utils.functions import mkdir_if_not_exists, seed_all
from src.utils.logger import log

DEFAULT_CID = "community"


def save_h5(state_dict, path):
    """Espelha o checkpoint em .h5, formato dos artefatos da entrega anterior."""
    with h5py.File(path, "w") as fh:
        for k, v in state_dict.items():
            fh.create_dataset(k, data=v.detach().cpu().numpy())


def main():
    parser = ArgumentParser()
    parser.add_argument("--loc", type=str, default="austin",
                        help="[austin, california, newyork, puertorico]")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--out_root", type=str, default="etc/in")
    parser.add_argument("--filter_bs", type=str, default=DEFAULT_CID,
                        help="cid do cliente, ou 'community' para a serie agregada")
    parser.add_argument("--identifier", type=str, default="cid")
    parser.add_argument("--targets", type=list, default=["consumption"])
    parser.add_argument("--num_lags", type=int, default=96)

    parser.add_argument("--criterion", type=str, default="mse")
    parser.add_argument("--model_name", type=str, default="lstm",
                        help='["rnn", "lstm", "gru", "cnn"]')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = f"dataset/pecanstreet/15min/{args.loc}/train/"
    if args.test_path is None:
        args.test_path = f"dataset/pecanstreet/15min/{args.loc}/test/"
    args.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    # A entidade treinada e sempre args.filter_bs: e ela que nomeia os .npy lidos
    # por ClientLearning e os artefatos escritos aqui.
    cid = str(args.filter_bs)

    log(INFO, f"Script arguments {args}")
    start = time.time()
    seed_all(args.seed)

    # A regiao entra no caminho de saida de proposito: o main_in.py antigo
    # escrevia em etc/in/<...>/<cid>_*, sem a regiao, e como ha cids repetidos
    # entre regioes rodar duas seguidas sobrescrevia a primeira.
    base = f"{args.out_root}/{args.loc}"
    results_dir = f"{base}/results/{args.model_name}"
    logs_dir = f"{base}/logs/{args.model_name}"
    ckpt_dir = f"{base}/ckpts/{args.model_name}/best"
    for d in (f"{results_dir}/preds/inverted", logs_dir, ckpt_dir):
        mkdir_if_not_exists(d)

    learner = ClientLearning(args=args, cid=cid, seed=args.seed)
    learner.prepare_model()
    train_loader, val_loader = learner._load_data()

    best_model, train_mse, val_mse = learner.train(
        train_loader=train_loader, val_loader=val_loader, model=learner.model,
        epochs=args.epochs, optimizer=args.optimizer, lr=args.lr,
        criterion=args.criterion, early_stopping=args.early_stopping,
        patience=args.patience, device=args.device,
    )

    np.save(f"{logs_dir}/{cid}_train_mse.npy", np.array(train_mse))
    np.save(f"{logs_dir}/{cid}_val_mse.npy", np.array(val_mse))

    T.save(best_model.state_dict(), f"{ckpt_dir}/{args.model_name}_{cid}.pth")
    save_h5(best_model.state_dict(), f"{ckpt_dir}/{args.model_name}_{cid}.h5")

    results, inverted = learner.test_model(best_model)

    def _py(v):
        """accumulate_metrics devolve np.float32/np.float64, que json nao serializa."""
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    def split(d):
        preds = {"y_true": d["y_true"], "y_pred": d["y_pred"], "client": d["client"]}
        metrics = {k: _py(v) for k, v in d.items() if k not in ("y_true", "y_pred")}
        return preds, metrics

    preds, metrics = split(results)
    inv_preds, inv_metrics = split(inverted)

    with open(f"{results_dir}/{cid}_results.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(f"{results_dir}/{cid}_results_inverted.json", "w") as fh:
        json.dump(inv_metrics, fh, indent=2)
    with open(f"{results_dir}/preds/{cid}_preds.json", "w") as fh:
        json.dump(preds, fh)
    with open(f"{results_dir}/preds/inverted/{cid}_preds.json", "w") as fh:
        json.dump(inv_preds, fh)

    # Criterio 6: o MSE do arquivo de metricas tem de ser reproduzivel a partir
    # do arquivo de previsoes.
    for tag, p, m in (("escalado", preds, metrics), ("invertido", inv_preds, inv_metrics)):
        recomputed = mean_squared_error(np.array(p["y_true"]), np.array(p["y_pred"]))
        assert np.isclose(recomputed, m["mse"], rtol=1e-6, atol=1e-12), (
            f"{tag}: MSE de results ({m['mse']}) nao reproduz o das preds ({recomputed})"
        )

    log(INFO, f"[{args.loc}/{args.model_name}/{cid}] melhor epoca de validacao = "
              f"{int(np.argmin(val_mse)) + 1}/{len(val_mse)} | "
              f"MSE teste = {metrics['mse']:.8f} | "
              f"MSE teste invertido = {inv_metrics['mse']:.6f}")
    log(INFO, f"Project time = {(time.time() - start) / 3600} hours")


if __name__ == "__main__":
    main()
