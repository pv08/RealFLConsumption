#!/usr/bin/env python
"""
tools/check_acceptance.py

Confere os seis criterios de aceitacao da secao 9 da instrucao de revisao.
Os criterios 1-4 (formato e conteudo dos community.csv) sao reexecutados por
tools/build_community_dataset.py::verify; aqui checamos 5 e 6, sobre os
artefatos de treino, e repetimos 1-4 para deixar uma saida unica.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.build_community_dataset import verify as verify_csv

MODELS = ["rnn", "lstm", "gru", "cnn"]


def check_artifacts(loc, out_root, model, epochs):
    base = f"{out_root}/{loc}"
    res = f"{base}/results/{model}"
    expected = [
        f"{res}/community_results.json",
        f"{res}/community_results_inverted.json",
        f"{res}/preds/community_preds.json",
        f"{res}/preds/inverted/community_preds.json",
        f"{base}/logs/{model}/community_train_mse.npy",
        f"{base}/logs/{model}/community_val_mse.npy",
        f"{base}/ckpts/{model}/best/{model}_community.h5",
        f"{base}/ckpts/{model}/best/{model}_community.pth",
    ]
    missing = [p for p in expected if not os.path.exists(p)]
    assert not missing, f"{loc}/{model}: faltam artefatos {missing}"

    for curve in ("train", "val"):
        arr = np.load(f"{base}/logs/{model}/community_{curve}_mse.npy")
        assert arr.shape == (epochs,), \
            f"{loc}/{model}: community_{curve}_mse.npy tem {arr.shape}, esperado ({epochs},)"

    out = {}
    for tag, rfile, pfile in (
        ("escalado", f"{res}/community_results.json", f"{res}/preds/community_preds.json"),
        ("invertido", f"{res}/community_results_inverted.json", f"{res}/preds/inverted/community_preds.json"),
    ):
        m = json.load(open(rfile))
        p = json.load(open(pfile))
        y_true, y_pred = np.array(p["y_true"]), np.array(p["y_pred"])
        assert y_true.shape == (96, 1), f"{loc}/{model}: y_true com shape {y_true.shape}, esperado (96, 1)"
        recomputed = mean_squared_error(y_true, y_pred)
        assert np.isclose(recomputed, m["mse"], rtol=1e-6, atol=1e-12), (
            f"{loc}/{model} ({tag}): MSE de results ({m['mse']}) nao reproduz o das preds ({recomputed})"
        )
        out[tag] = m

    val = np.load(f"{base}/logs/{model}/community_val_mse.npy")
    return out, int(np.argmin(val)) + 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset/pecanstreet/15min")
    ap.add_argument("--manifest-dir", default="etc/community")
    ap.add_argument("--out-root", default="etc/in")
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    locs = sorted(os.path.basename(p).replace("_community.json", "")
                  for p in glob.glob(f"{args.manifest_dir}/*_community.json"))

    print("== Criterios 1-4: series comunitarias ==")
    for loc in locs:
        verify_csv(loc, args.root, args.manifest_dir)

    print("\n== Criterios 5-6: artefatos de treino ==")
    print(f"{'regiao':12s} {'modelo':6s} {'melhor ep':>9s} {'MSE (escalado)':>16s} {'MSE (kW^2)':>12s} {'RMSE (kW)':>10s}")
    for loc in locs:
        man = json.load(open(f"{args.manifest_dir}/{loc}_community.json"))
        for m in MODELS:
            metrics, best_ep = check_artifacts(loc, args.out_root, m, args.epochs)
            print(f"{loc:12s} {m:6s} {best_ep:9d} {metrics['escalado']['mse']:16.8f} "
                  f"{metrics['invertido']['mse']:12.4f} {metrics['invertido']['rmse']:10.4f}")
        print(f"{'':12s} (n={man['n_community']}, alvo medio "
              f"{man['mean_consumption_test_target']:.3f} kW)")

    print("\nOK: os seis criterios passaram.")
