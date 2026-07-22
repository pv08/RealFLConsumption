#!/bin/bash

# Parse --loc / -loc, --epochs / -epochs, --clients_per_round / -clients_per_round, --max_rounds / -max_rounds, --gpu_slots / -gpu_slots, --notify / -notify, --wandb / -wandb
TARGET_LOC=""
EPOCHS=200
CLIENTS_PER_ROUND=5
MAX_ROUNDS=10
GPU_SLOTS=1
NOTIFY=false
WANDB=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --loc|-loc)
            TARGET_LOC="$2"
            shift 2
            ;;
        --epochs|-epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --clients_per_round|-clients_per_round)
            CLIENTS_PER_ROUND="$2"
            shift 2
            ;;
        --max_rounds|-max_rounds)
            MAX_ROUNDS="$2"
            shift 2
            ;;
        --gpu_slots|-gpu_slots)
            GPU_SLOTS="$2"
            shift 2
            ;;
        --notify|-notify)
            NOTIFY=true
            shift
            ;;
        --wandb|-wandb)
            WANDB=true
            shift
            ;;
        *)
            echo "[!] - Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Notifications are opt-in (--notify). When enabled, NOTIFY_WEBHOOK_URL must be
# set in .env for both this script and the containerized fl-server to pick up.
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
    echo "[!] - Usage: bash $0 -loc <austin|california|newyork|puertorico> [-epochs <n>] [-clients_per_round <n>] [-max_rounds <n>] [-gpu_slots <n>] [-notify] [-wandb]"
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
echo "[!] - Epochs: $EPOCHS | Clients per round: $CLIENTS_PER_ROUND | Max rounds: $MAX_ROUNDS | GPU slots: $GPU_SLOTS"

# Seed pattern: <fraction><epochs><fl_rounds> concatenated (fraction=clients_per_round,
# fl_rounds=max_rounds), plus the model index so each model keeps a distinct seed.
# e.g. clients_per_round=5, epochs=200, max_rounds=10 -> base 520010 -> rnn=520010, lstm=520011, gru=520012.
MODELS=(rnn lstm gru)
BASE_SEED="${CLIENTS_PER_ROUND}${EPOCHS}${MAX_ROUNDS}"
echo "[!] - Base seed (<clients_per_round><epochs><max_rounds>): $BASE_SEED"
TOTAL=${#MODELS[@]}
RESULTS=()
NOTIFY_ARG=""
if [[ "$NOTIFY" == "true" ]]; then
    NOTIFY_ARG="--enable_notifications"
fi
WANDB_ARG=""
if [[ "$WANDB" == "true" ]]; then
    WANDB_ARG="--enable_wandb"
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
    SEED=$((BASE_SEED + (i + 1)))
    CURRENT=$((i + 1))
    COMPOSE_FILE="docker-compose.gpu.${MODEL}.${TARGET_LOC}.seed${SEED}.yml"

    echo "------------------------------------------------"
    echo "[$CURRENT/$TOTAL] - Generating simulation for model=${MODEL} loc=${TARGET_LOC} seed=${SEED}"

    python generate_simulation.py --loc "$TARGET_LOC" --model_name "$MODEL" --seed "$SEED" \
        --epochs "$EPOCHS" --clients_per_round "$CLIENTS_PER_ROUND" --max_rounds "$MAX_ROUNDS" \
        --gpu_slots "$GPU_SLOTS" $NOTIFY_ARG $WANDB_ARG

    if [[ ! -f "$COMPOSE_FILE" ]]; then
        echo "[!] - Expected compose file ${COMPOSE_FILE} was not generated. Skipping..."
        continue
    fi

    echo "[$CURRENT/$TOTAL] - Running docker compose: ${COMPOSE_FILE}"
    CURRENT_COMPOSE_FILE="$COMPOSE_FILE"

    # --exit-code-from fl-server waits for the server (which only exits once
    # every client has returned its final test results) instead of aborting
    # as soon as the first client container exits.
    docker compose -f "$COMPOSE_FILE" up --build --exit-code-from fl-server

    if [ $? -eq 0 ]; then
        echo "[*] - Finished model ${MODEL} (seed ${SEED})."
        RESULTS+=("${MODEL}: OK")
    else
        echo "[!] - Simulation failed for model ${MODEL} (seed ${SEED}). Starting the next..."
        RESULTS+=("${MODEL}: FAILED")
        notify "FL batch (${TARGET_LOC}): model ${MODEL} (seed ${SEED}) FAILED. Continuing with the next model."
    fi

    echo "[!] - Tearing down ${COMPOSE_FILE}..."
    docker compose -f "$COMPOSE_FILE" down
    unset CURRENT_COMPOSE_FILE

    echo "------------------------------------------------"
done

echo "[!] - All models finished for location: ${TARGET_LOC}"
SUMMARY="FL batch finished for ${TARGET_LOC}: $(printf '%s | ' "${RESULTS[@]}")"
notify "$SUMMARY"
