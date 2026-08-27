#!/usr/bin/env bash
# Roda o lote de treino centralizado dentro do container, repassando os
# argumentos para o run_centralized.sh.
#
# Todos os argumentos do runner valem aqui:
#   bash run_centralized_docker.sh -loc all -bootstrap -jobs 3
#   bash run_centralized_docker.sh -loc austin -models "rnn lstm gru" -epochs 10
#   bash run_centralized_docker.sh -scope community -batch_size 512
#   bash run_centralized_docker.sh -loc austin -dry-run
#
# Acompanhe pelo host, normalmente: os logs por job caem em etc/, que e volume.
#   bash progress.sh -w
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

COMPOSE_FILE=docker-compose.centralized.yml
BUILD="${BUILD:-1}"

# A imagem embute o codigo (COPY . .), entao um script alterado no host so
# chega no container depois do build. A camada do COPY e a ultima, e o rebuild
# custa segundos quando so o codigo mudou.
if [[ "$BUILD" == "1" ]]; then
    echo "[!] - Atualizando a imagem (BUILD=0 para pular)..."
    docker compose -f "$COMPOSE_FILE" build centralized
fi

# `run --rm` em vez de `up`: sem container orfao, e o codigo de saida do runner
# vira o codigo de saida daqui, o que deixa encadear em script.
exec docker compose -f "$COMPOSE_FILE" run --rm centralized \
    bash run_centralized.sh "$@"
