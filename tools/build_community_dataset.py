#!/usr/bin/env python
"""
tools/build_community_dataset.py

Reconstroi a serie comunitaria de cada regiao a partir dos arquivos ja divididos
em dataset/pecanstreet/15min/<loc>/{train,test}/<cid>.csv.

Substitui _gen_community_dataset (celula 5 de pecanstreet_divider.ipynb), que
tinha quatro defeitos:
  1. inicializava o acumulador com train_files[0]/test_files[0] e depois iterava
     a partir do indice 0, somando o primeiro cliente DUAS vezes, no treino e no
     alvo;
  2. somava por posicao de linha, nao por carimbo de tempo;
  3. usava min_length = min(len(train_df), len(train_df)), que compara o
     dataframe consigo mesmo;
  4. mantinha leg1v/leg2v do primeiro cliente em vez de media (tensao nao e
     grandeza somavel).

Aqui a soma e feita por carimbo de tempo, cada cliente entra exatamente uma vez,
e a janela de teste da comunidade e forcada a ser identica a dos clientes, de
modo que o agregado federado e o modelo comunitario passem a prever a mesma
grandeza, no mesmo dia.

Esquema real dos CSV desta base: 46 colunas, com uma unica coluna `generation`
(nao existem generation_solar1/generation_solar2).

Uso:
    python tools/build_community_dataset.py --loc all --dry-run   # so diagnostica
    python tools/build_community_dataset.py --loc austin          # escreve os CSV
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.functions import get_available_clients_location

LOCATIONS = ["austin", "california", "newyork", "puertorico"]

# Grandezas extensivas: somam entre domicilios.
SUM_COLS = ["consumption", "generation"]
# Grandezas intensivas: tensao nao soma, tira-se a media.
MEAN_COLS = ["leg1v", "leg2v"]
# Derivadas da serie somada, recalculadas depois da soma.
DERIVED_COLS = ["prev_consumption", "consumption_change"]

TEST_WINDOW = 192   # 96 de contexto + 96 de alvo
CONTEXT = 96


def _read(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    return (df.drop_duplicates(subset="Date", keep="first")
              .sort_values("Date")
              .reset_index(drop=True))


def survey(loc, root):
    """Le todos os clientes da regiao e decide a composicao da comunidade."""
    base = f"{root}/{loc}"
    cids = list(get_available_clients_location(loc))
    train = {c: _read(f"{base}/train/{c}.csv") for c in cids}
    test = {c: _read(f"{base}/test/{c}.csv") for c in cids}

    # A comunidade so faz sentido se os clientes preveem o mesmo dia. Mantem-se a
    # janela de teste modal; quem diverge sai do agregado (segue na analise
    # por cliente).
    windows = {c: tuple(test[c]["Date"]) for c in cids}
    modal, _ = collections.Counter(windows.values()).most_common(1)[0]
    keep = [c for c in cids if windows[c] == modal]
    drop = [c for c in cids if windows[c] != modal]

    # Periodo comum de treino: interseccao EXATA de carimbos entre os mantidos.
    # Sem reindexar para a grade cheia e sem preencher lacuna com zero.
    common = None
    for c in keep:
        s = set(train[c]["Date"])
        common = s if common is None else (common & s)
    common_train = pd.DatetimeIndex(sorted(common), name="Date")

    # Massa de NaN de consumption por cliente sobre o periodo comum: entra com 0
    # na soma, entao vale registrar quanto isso representa.
    nan_share = {}
    for c in keep:
        v = train[c].set_index("Date").reindex(common_train)["consumption"]
        nan_share[str(c)] = round(float(v.isna().mean()), 6)
    rows_with_gap = np.zeros(len(common_train), dtype=bool)
    for c in keep:
        v = train[c].set_index("Date").reindex(common_train)["consumption"]
        rows_with_gap |= v.isna().to_numpy()

    span = common_train[-1] - common_train[0]
    report = {
        "location": loc,
        "n_total": len(cids),
        "n_community": len(keep),
        "clients": keep,
        "excluded": drop,
        "excluded_windows": {
            str(c): [str(windows[c][CONTEXT]), str(windows[c][-1])] for c in drop
        },
        "distinct_test_windows": len(set(windows.values())),
        "test_context_start": str(modal[0]),
        "test_target_start": str(modal[CONTEXT]),
        "test_target_end": str(modal[-1]),
        "train_common_start": str(common_train[0]),
        "train_common_end": str(common_train[-1]),
        "train_common_rows": int(len(common_train)),
        "train_span_days": int(span.days),
        "train_grid_coverage": round(
            len(common_train) / (int(span.total_seconds() // 900) + 1), 4
        ),
        "nan_policy": "fillna(0) por cliente antes da soma",
        "train_consumption_nan_share_per_client": nan_share,
        "train_rows_with_any_nan": int(rows_with_gap.sum()),
    }
    return train, test, keep, pd.DatetimeIndex(modal, name="Date"), common_train, report


def build(loc, root, manifest_dir):
    train, test, keep, test_index, train_index, report = survey(loc, root)
    base = f"{root}/{loc}"

    # Serie completa por cliente (treino + teste), indexada por carimbo. O treino
    # termina na ultima linha de contexto, entao os dois blocos se encaixam.
    full = {}
    for c in keep:
        d = pd.concat([train[c], test[c]], ignore_index=True)
        full[c] = (d.drop_duplicates(subset="Date", keep="first")
                    .set_index("Date").sort_index())

    index = train_index.union(test_index)

    # Clima e calendario vem do cliente de referencia: verificado coluna a coluna
    # que sao identicos entre clientes no mesmo instante.
    ref_cid = keep[0]
    columns = list(train[ref_cid].columns)
    community = full[ref_cid].reindex(index).copy()

    for col in SUM_COLS:
        acc = np.zeros(len(index))
        for c in keep:
            acc += (full[c].reindex(index)[col]
                          .astype("float64").fillna(0.0).to_numpy())
        community[col] = acc

    for col in MEAN_COLS:
        stack = np.vstack([
            full[c].reindex(index)[col].astype("float64").to_numpy() for c in keep
        ])
        with np.errstate(invalid="ignore"):
            community[col] = np.nanmean(stack, axis=0)

    community = community.reset_index()
    community["prev_consumption"] = community["consumption"].shift(1)
    community["consumption_change"] = (
        community["consumption"] - community["prev_consumption"]
    ).fillna(0.0)
    community["cid"] = "community"
    community = community[columns]

    train_df = community[community["Date"].isin(train_index)].reset_index(drop=True)
    test_df = community[community["Date"].isin(test_index)].reset_index(drop=True)

    assert len(test_df) == TEST_WINDOW, f"{loc}: janela de teste com {len(test_df)} linhas"
    assert list(test_df["Date"]) == list(test_index), f"{loc}: janela de teste desalinhada"
    assert len(train_df) == len(train_index), f"{loc}: treino com {len(train_df)} linhas"

    train_df.to_csv(f"{base}/train/community.csv", index_label=False)
    test_df.to_csv(f"{base}/test/community.csv", index_label=False)

    report["reference_client"] = ref_cid
    report["n_columns"] = len(columns)
    report["train_rows_written"] = int(len(train_df))
    report["test_rows_written"] = int(len(test_df))
    report["mean_consumption_train"] = float(train_df["consumption"].mean())
    report["mean_consumption_test_context"] = float(test_df["consumption"].iloc[:CONTEXT].mean())
    report["mean_consumption_test_target"] = float(test_df["consumption"].iloc[CONTEXT:].mean())
    os.makedirs(manifest_dir, exist_ok=True)
    with open(f"{manifest_dir}/{loc}_community.json", "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def verify(loc, root, manifest_dir="etc/community"):
    """Criterios 1-4 da secao 9 da instrucao de revisao."""
    base = f"{root}/{loc}"
    manifest = json.load(open(f"{manifest_dir}/{loc}_community.json"))
    keep = manifest["clients"]
    ref = manifest["reference_client"]

    comm_train = _read(f"{base}/train/community.csv")
    comm_test = _read(f"{base}/test/community.csv")

    # 1. cabecalho identico, nome e ordem, ao de um arquivo de cliente
    client_cols = list(_read(f"{base}/train/{ref}.csv").columns)
    assert list(comm_train.columns) == client_cols, f"{loc}: cabecalho de treino difere do cliente"
    assert list(comm_test.columns) == client_cols, f"{loc}: cabecalho de teste difere do cliente"

    # 2. 192 linhas de teste, com as mesmas datas de qualquer cliente mantido
    assert len(comm_test) == TEST_WINDOW, f"{loc}: teste com {len(comm_test)} linhas"
    for c in keep:
        cd = list(_read(f"{base}/test/{c}.csv")["Date"])
        assert list(comm_test["Date"]) == cd, f"{loc}/{c}: janela de teste divergente"

    # 3. o alvo e exatamente a soma dos mantidos, e nenhum cliente esta duplicado
    tgt = comm_test.iloc[CONTEXT:].set_index("Date")["consumption"]
    total = np.zeros(len(tgt))
    parcels = {}
    for c in keep:
        d = _read(f"{base}/test/{c}.csv").set_index("Date")
        v = d.reindex(tgt.index)["consumption"].astype("float64")
        assert v.notna().all(), f"{loc}/{c}: faltam carimbos na janela de teste"
        parcels[c] = v.to_numpy()
        total += parcels[c]

    err = float(np.abs(total - tgt.to_numpy()).max())
    dup_cid, dup = min(
        ((c, float(np.abs(total + v - tgt.to_numpy()).max())) for c, v in parcels.items()),
        key=lambda kv: kv[1],
    )

    # 4. Date de treino estritamente crescente, sem repeticao, e presente em todos
    d = comm_train["Date"]
    assert d.is_monotonic_increasing and d.is_unique, f"{loc}: Date de treino nao e estritamente crescente"
    tset = set(d)
    for c in keep:
        cd = set(_read(f"{base}/train/{c}.csv")["Date"])
        assert tset <= cd, f"{loc}/{c}: treino comunitario tem carimbo ausente no cliente"

    print(f"[{loc}] n={len(keep)}  |soma - alvo|max = {err:.3e}   "
          f"melhor hipotese de duplicata = {dup:.4f} (cid {dup_cid}, tem de ser >> 0)")
    assert err < 1e-9, f"{loc}: alvo comunitario nao bate com a soma dos clientes"
    assert dup > 1e-6, f"{loc}: ainda ha um cliente duplicado"
    return err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", default="all")
    ap.add_argument("--root", default="dataset/pecanstreet/15min")
    ap.add_argument("--manifest-dir", default="etc/community")
    ap.add_argument("--dry-run", action="store_true", help="so diagnostica, nao escreve CSV")
    args = ap.parse_args()

    locs = LOCATIONS if args.loc == "all" else [args.loc]
    for loc in locs:
        if args.dry_run:
            *_, report = survey(loc, args.root)
            print(json.dumps(report, indent=2))
        else:
            print(json.dumps(build(loc, args.root, args.manifest_dir), indent=2))
            verify(loc, args.root, args.manifest_dir)
