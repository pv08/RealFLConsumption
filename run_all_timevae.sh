#!/bin/bash

# Roda a avaliacao generativa do TimeVAE (TSTR/TRTS) para TODOS os clientes de uma localidade,
# um modelo de cada vez. Espelha o run_all_models.sh, que faz o mesmo para a simulacao FL.
#
#   bash run_all_timevae.sh -loc austin -gpu_slots 3
#
# Cada modelo gera o seu docker-compose (generate_timevae_experiments.py) e sobe todos os
# clientes juntos; quem limita a GPU e o GPULock, via -gpu_slots.

TARGET_LOC=""
MODELS=(rnn lstm gru)
ARMS=(A6)
SEEDS=(0)
GPU_SLOTS=1
TIMEVAE_EPOCHS=200
R_EPOCHS=20
GEN_HOLDOUT_DAYS=0
PLOTS=true
NOTIFY=false
MODE="both"
R_BATCH_SIZE=""
BATCH_SIZE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --loc|-loc)
            TARGET_LOC="$2"
            shift 2
            ;;
        --models|-models)
            # Aceita lista: -models lstm gru
            shift
            MODELS=()
            while [[ $# -gt 0 && "$1" != -* ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --arms|-arms)
            shift
            ARMS=()
            while [[ $# -gt 0 && "$1" != -* ]]; do
                ARMS+=("$1")
                shift
            done
            ;;
        --seeds|-seeds)
            shift
            SEEDS=()
            while [[ $# -gt 0 && "$1" != -* ]]; do
                SEEDS+=("$1")
                shift
            done
            ;;
        --gpu_slots|-gpu_slots)
            GPU_SLOTS="$2"
            shift 2
            ;;
        --timevae_epochs|-timevae_epochs)
            TIMEVAE_EPOCHS="$2"
            shift 2
            ;;
        --r_epochs|-r_epochs)
            R_EPOCHS="$2"
            shift 2
            ;;
        --gen_holdout_days|-gen_holdout_days)
            GEN_HOLDOUT_DAYS="$2"
            shift 2
            ;;
        --no_plots|-no_plots)
            PLOTS=false
            shift
            ;;
        --notify|-notify)
            NOTIFY=true
            shift
            ;;
        --mode|-mode)
            MODE="$2"
            shift 2
            ;;
        --r_batch_size|-r_batch_size)
            R_BATCH_SIZE="$2"
            shift 2
            ;;
        --batch_size|-batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "[!] - Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Notifications are opt-in (--notify). When enabled, NOTIFY_WEBHOOK_URL must be
# set in .env for this script to pick up.
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

notify() {
    if [[ "$NOTIFY" == "true" ]]; then
        python3 -m src.utils.notifier "$1"
    fi
}

if [[ -z "$TARGET_LOC" ]]; then
    echo "[!] - Usage: bash $0 -loc <austin|california|newyork|puertorico> [-models rnn lstm gru]"
    echo "            [-arms A6] [-seeds 0] [-gpu_slots <n>] [-timevae_epochs <n>] [-r_epochs <n>]"
    echo "            [-gen_holdout_days <n>] [-no_plots] [-notify]"
    exit 1
fi

case "$TARGET_LOC" in
    austin|california|newyork|puertorico) ;;
    *)
        echo "[!] - Unknown location '$TARGET_LOC'. Valid options: austin, california, newyork, puertorico"
        exit 1
        ;;
esac

echo "[!] - Target location: $TARGET_LOC"
echo "[!] - Models: ${MODELS[*]} | Arms: ${ARMS[*]} | Seeds: ${SEEDS[*]}"
echo "[!] - GPU slots: $GPU_SLOTS | TimeVAE epochs: $TIMEVAE_EPOCHS | Regressor epochs: $R_EPOCHS"

TOTAL=${#MODELS[@]}
RESULTS=()
HOLDOUT_ARG=""
if [[ "$GEN_HOLDOUT_DAYS" -gt 0 ]]; then
    HOLDOUT_ARG="--gen_holdout_days $GEN_HOLDOUT_DAYS"
fi
PLOTS_ARG=""
if [[ "$PLOTS" == "false" ]]; then
    PLOTS_ARG="--no_plots"
fi
R_BATCH_SIZE_ARG=""
if [[ -n "$R_BATCH_SIZE" ]]; then
    R_BATCH_SIZE_ARG="--r_batch_size $R_BATCH_SIZE"
fi
BATCH_SIZE_ARG=""
if [[ -n "$BATCH_SIZE" ]]; then
    BATCH_SIZE_ARG="--batch_size $BATCH_SIZE"
fi

# Safety net: if the script gets interrupted mid-run, still tear down
# whichever compose stack is currently up so container names are freed.
cleanup() {
    if [[ -n "${CURRENT_COMPOSE_FILE:-}" ]]; then
        echo "[!] - Cleaning up ${CURRENT_COMPOSE_FILE}..."
        docker compose -f "$CURRENT_COMPOSE_FILE" down
    fi
}
trap cleanup EXIT

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    CURRENT=$((i + 1))
    COMPOSE_FILE="docker-compose.timevae.${TARGET_LOC}.${MODEL}.yml"

    echo "------------------------------------------------"
    echo "[$CURRENT/$TOTAL] - Generating TimeVAE experiments for model=${MODEL} loc=${TARGET_LOC}"

    python generate_timevae_experiments.py --loc "$TARGET_LOC" --model_name "$MODEL" \
        --arms "${ARMS[@]}" --seeds "${SEEDS[@]}" --gpu_slots "$GPU_SLOTS" \
        --timevae_epochs "$TIMEVAE_EPOCHS" --r_epochs "$R_EPOCHS" \
        --mode "$MODE" \
        --out "$COMPOSE_FILE" $HOLDOUT_ARG $PLOTS_ARG $R_BATCH_SIZE_ARG $BATCH_SIZE_ARG

    if [[ ! -f "$COMPOSE_FILE" ]]; then
        echo "[!] - Expected compose file ${COMPOSE_FILE} was not generated. Skipping..."
        continue
    fi

    echo "[$CURRENT/$TOTAL] - Running docker compose: ${COMPOSE_FILE}"
    CURRENT_COMPOSE_FILE="$COMPOSE_FILE"

    # Sem --exit-code-from / --abort-on-container-exit: aqui nao existe um servico "coordenador"
    # como o fl-server, sao N clientes irmaos, e qualquer uma dessas flags derrubaria os demais
    # assim que o primeiro terminasse. `up` volta quando todos pararem.
    docker compose -f "$COMPOSE_FILE" up --build

    # `up` sai com 0 mesmo que um cliente tenha falhado, entao o veredito vem dos codigos de
    # saida individuais.
    FAILED=$(docker compose -f "$COMPOSE_FILE" ps -aq \
             | xargs -r docker inspect -f '{{.Name}} {{.State.ExitCode}}' \
             | awk '$2 != 0 { print $1 }')
    N_FAILED=$(echo -n "$FAILED" | grep -c . )

    if [[ "$N_FAILED" -eq 0 ]]; then
        echo "[*] - Finished model ${MODEL} (all clients OK)."
        RESULTS+=("${MODEL}: OK")
    else
        echo "[!] - ${N_FAILED} container(s) failed for model ${MODEL}:"
        echo "$FAILED" | sed 's/^/        /'
        RESULTS+=("${MODEL}: FAILED (${N_FAILED})")
        notify "TimeVAE batch (${TARGET_LOC}): model ${MODEL} had ${N_FAILED} failed client(s). Continuing with the next model."
    fi

    echo "[!] - Tearing down ${COMPOSE_FILE}..."
    docker compose -f "$COMPOSE_FILE" down
    unset CURRENT_COMPOSE_FILE

    echo "------------------------------------------------"
done

echo "[!] - All models finished for location: ${TARGET_LOC}"
SUMMARY="TimeVAE batch finished for ${TARGET_LOC}: $(printf '%s | ' "${RESULTS[@]}")"
echo "[!] - $SUMMARY"
notify "$SUMMARY"
