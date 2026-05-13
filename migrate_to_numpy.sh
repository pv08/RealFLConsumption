#!/bin/bash

# Parse --loc argument
TARGET_LOC=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --loc)
            TARGET_LOC="$2"
            shift 2
            ;;
        *)
            echo "[!] - Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$TARGET_LOC" ]]; then
    echo "[!] - Usage: bash $0 --loc <austin|ny|california|puerto_rico>"
    exit 1
fi

echo "[!] - Initiating migration from CSV to Numpy"
echo "[!] - Target location: $TARGET_LOC"

# Lista de IDs (Copiada do seu entrypoint)
AUSTIN_LIST=(8156 2335 9922 3039 4031 8386 7951 3538 9160 5746 1642 6139 2361 3456 9019 661 2818 7800 7536 4767 8565 4373 7901 7719 9278)
NY_LIST=(1417 142 3000 3996 4550 2096 5058 387 2318 5997 5982 914 4283 3700 3517 3488 27 5587 1240 2358 558 5679 1222 950 9053)
CALIFORNIA_LIST=(8342 7114 7062 6547 8733 9612 9836 9775 4495 9213 4934 2606 8061 3687 1731 3938 1450 1524 3864 203 5938 6377 8574)
PUERTO_RICO_LIST=(8235 3835 7667 3987 11386 10887 6056 939 4046 11126 10826 1092 1286 1245 7778 2006 2159 11933 2921 8997 371 9911 4204 4749 6406)

case "$TARGET_LOC" in
    austin)       ID_LIST=("${AUSTIN_LIST[@]}") ;;
    new_york)     ID_LIST=("${NY_LIST[@]}") ;;
    california)   ID_LIST=("${CALIFORNIA_LIST[@]}") ;;
    puerto_rico)  ID_LIST=("${PUERTO_RICO_LIST[@]}") ;;
    *)
        echo "[!] - Unknown location '$TARGET_LOC'. Valid options: austin, ny, california, puerto_rico"
        exit 1
        ;;
esac

# Contadores para feedback visual
TOTAL=${#ID_LIST[@]}
CURRENT=0

echo "Clients available to process: $TOTAL"
echo "------------------------------------------------"

# Loop para iterar sobre cada ID e chamar o script Python
for CLIENT_ID in "${ID_LIST[@]}"; do
    CURRENT=$((CURRENT + 1))

    echo "[$CURRENT/$TOTAL] Processing Cliente ID: $CLIENT_ID ..."

    # Chama o script Python worker passando o ID e a Localidade
    python migrate_data_numpy.py --loc $TARGET_LOC --filter_bs "$CLIENT_ID" --data_path "dataset/pecanstreet/15min/${TARGET_LOC}/train/" --test_path "dataset/pecanstreet/15min/${TARGET_LOC}/test/"\


    # Verifica se o python rodou com sucesso (Exit Code 0)
    if [ $? -eq 0 ]; then
        echo "[*] - Finished."
    else
        echo " [!] - Fail to migrate client $CLIENT_ID. Starting the next..."
    fi

    echo "------------------------------------------------"
done

echo "[!] - Migration finished!"