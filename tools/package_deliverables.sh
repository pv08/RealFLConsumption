#!/usr/bin/env bash
# Monta a arvore de entregaveis da secao 10 da instrucao de revisao.
set -euo pipefail

OUT="${OUT:-sim-results/in_community_fix}"
ROOT="${ROOT:-dataset/pecanstreet/15min}"

mkdir -p "$OUT"

# 1. community.csv das regioes reconstruidas
for LOC in austin california newyork puertorico; do
  mkdir -p "$OUT/$LOC/dataset/train" "$OUT/$LOC/dataset/test"
  cp "$ROOT/$LOC/train/community.csv" "$OUT/$LOC/dataset/train/"
  cp "$ROOT/$LOC/test/community.csv"  "$OUT/$LOC/dataset/test/"
done

# 2. manifestos: a lista de clientes, os excluidos, as janelas e o periodo comum.
#    E com este arquivo que o lado federado sera reagregado sobre exatamente os
#    mesmos clientes e o mesmo dia.
mkdir -p "$OUT/manifests"
cp etc/community/*_community.json "$OUT/manifests/"

# 3. arvore completa das execucoes novas, por regiao
for LOC in austin california newyork puertorico; do
  if [ -d "etc/in/$LOC" ]; then
    rm -rf "$OUT/$LOC/in"
    mkdir -p "$OUT/$LOC"
    cp -r "etc/in/$LOC" "$OUT/$LOC/in"
  fi
done

# 4. o script na versao que rodou
mkdir -p "$OUT/tools"
cp tools/build_community_dataset.py tools/check_acceptance.py "$OUT/tools/"
cp train_community.py run_community.sh "$OUT/tools/"

# (dry_run.log, build.log e train.log ja vivem em $OUT)
echo "[!] Entregaveis em $OUT"
find "$OUT" -maxdepth 2 -type d | sort
