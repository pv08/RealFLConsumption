#!/bin/bash
set -euo pipefail

# Roda a mesma simulação (via generate_simulation.py + docker compose) para uma lista de
# seeds, sequencialmente, movendo etc/fl/ para etc/fl_runs/seed_<n>/fl/ entre execuções
# para não sobrescrever os resultados da seed anterior.
#
# Uso:
#   bash run_seed_sweep.sh --loc austin --model_name rnn --seeds "1 2 3 4 5" \
#       --max_rounds 10 --epochs 200 --clients_per_round 5 [--optimize_clients] [--disable_blockchain]
#
# Leitura de resultados depois: iterar etc/fl_runs/seed_<n>/fl/... por seed, em vez de
# um único etc/fl/.

LOC=""
MODEL_NAME=""
SEEDS=""
MAX_ROUNDS=10
EPOCHS=200
CLIENTS_PER_ROUND=5
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --loc) LOC="$2"; shift 2 ;;
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --max_rounds) MAX_ROUNDS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --clients_per_round) CLIENTS_PER_ROUND="$2"; shift 2 ;;
        --optimize_clients) EXTRA_FLAGS="$EXTRA_FLAGS --optimize_clients"; shift ;;
        --disable_blockchain) EXTRA_FLAGS="$EXTRA_FLAGS --disable_blockchain"; shift ;;
        *) echo "[!] - Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$LOC" || -z "$MODEL_NAME" || -z "$SEEDS" ]]; then
    echo "[!] - Usage: bash $0 --loc <austin|newyork|california|puertorico> --model_name <rnn|lstm|gru|cnn> --seeds \"1 2 3\" [--max_rounds N] [--epochs N] [--clients_per_round N] [--optimize_clients] [--disable_blockchain]"
    exit 1
fi

echo "[!] - Seed sweep: loc=$LOC model=$MODEL_NAME seeds=($SEEDS) max_rounds=$MAX_ROUNDS epochs=$EPOCHS clients_per_round=$CLIENTS_PER_ROUND"

for SEED in $SEEDS; do
    echo "------------------------------------------------"
    echo "[!] - Running seed $SEED"

    COMPOSE_FILE="docker-compose.gpu.${MODEL_NAME}.${LOC}.seed${SEED}.yml"

    python generate_simulation.py --loc "$LOC" --model_name "$MODEL_NAME" --seed "$SEED" \
        --max_rounds "$MAX_ROUNDS" --epochs "$EPOCHS" --clients_per_round "$CLIENTS_PER_ROUND" $EXTRA_FLAGS

    docker compose -f "$COMPOSE_FILE" up --build --abort-on-container-exit
    docker compose -f "$COMPOSE_FILE" down

    if [ -d "etc/fl" ]; then
        mkdir -p "etc/fl_runs/seed_${SEED}"
        mv etc/fl "etc/fl_runs/seed_${SEED}/fl"
        echo "[*] - Results archived to etc/fl_runs/seed_${SEED}/fl"
    else
        echo "[!] - No etc/fl output found for seed $SEED. Skipping archive."
    fi
done

echo "------------------------------------------------"
echo "[!] - Seed sweep finished. Results under etc/fl_runs/seed_<n>/fl/"
