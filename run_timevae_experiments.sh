#!/bin/bash

# Parse arguments
TARGET_LOC=""
MODELS_STR=""
SEEDS="0 1 2"
TIMEVAE_EPOCHS=""
R_EPOCHS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --loc|-loc)
            TARGET_LOC="$2"
            shift 2
            ;;
        --models|-models)
            MODELS_STR="$2"
            shift 2
            ;;
        --seeds|-seeds)
            SEEDS="$2"
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
        *)
            echo "[!] - Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$TARGET_LOC" || -z "$MODELS_STR" ]]; then
    echo "[!] - Usage: bash $0 -loc <austin|california|newyork|puertorico> -models \"<model1> [model2 ...]\" [-seeds \"0 1 2\"] [-timevae_epochs <n>] [-r_epochs <n>]"
    exit 1
fi

case "$TARGET_LOC" in
    austin|california|newyork|puertorico) ;;
    *)
        echo "[!] - Unknown location '$TARGET_LOC'. Valid options: austin, california, newyork, puertorico"
        exit 1
        ;;
esac

# Convert MODELS_STR into an array
read -ra MODELS <<< "$MODELS_STR"

echo "[!] - Target location: $TARGET_LOC"
echo "[!] - Models to run: ${MODELS[@]}"
echo "[!] - Seeds: $SEEDS"

# Safety net: if the script gets interrupted mid-run, tear down the current compose stack
cleanup() {
    if [[ -n "${CURRENT_COMPOSE_FILE:-}" ]]; then
        echo "[!] - Cleaning up ${CURRENT_COMPOSE_FILE}..."
        docker compose -f "$CURRENT_COMPOSE_FILE" down
    fi
}
trap cleanup EXIT

TOTAL=${#MODELS[@]}
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    CURRENT=$((i + 1))
    
    COMPOSE_FILE="docker-compose.timevae.${MODEL}.${TARGET_LOC}.yml"

    echo "------------------------------------------------"
    echo "[$CURRENT/$TOTAL] - Generating TimeVAE simulation for model=${MODEL} loc=${TARGET_LOC}"

    # Build python arguments
    PY_ARGS=("--loc" "$TARGET_LOC" "--model_name" "$MODEL" "--arms" "A6" "--out" "$COMPOSE_FILE" "--seeds" $SEEDS)
    
    if [[ -n "$TIMEVAE_EPOCHS" ]]; then
        PY_ARGS+=("--timevae_epochs" "$TIMEVAE_EPOCHS")
    fi
    if [[ -n "$R_EPOCHS" ]]; then
        PY_ARGS+=("--r_epochs" "$R_EPOCHS")
    fi

    # Generate yaml
    python generate_timevae_experiments.py "${PY_ARGS[@]}"

    if [[ ! -f "$COMPOSE_FILE" ]]; then
        echo "[!] - Expected compose file ${COMPOSE_FILE} was not generated. Skipping..."
        continue
    fi

    echo "[$CURRENT/$TOTAL] - Running docker compose: ${COMPOSE_FILE}"
    CURRENT_COMPOSE_FILE="$COMPOSE_FILE"

    # Run docker compose
    docker compose -f "$COMPOSE_FILE" up --build

    if [ $? -eq 0 ]; then
        echo "[*] - Finished model ${MODEL}."
    else
        echo "[!] - Simulation failed for model ${MODEL}. Starting the next..."
    fi

    echo "[!] - Tearing down ${COMPOSE_FILE}..."
    docker compose -f "$COMPOSE_FILE" down
    unset CURRENT_COMPOSE_FILE

    echo "------------------------------------------------"
done

echo "[!] - All models finished for location: ${TARGET_LOC}"
