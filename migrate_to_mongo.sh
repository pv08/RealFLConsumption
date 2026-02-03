#!/bin/bash

# Configura a localidade
TARGET_LOC="austin"

echo "[!] - Initiating migration from CSV to MongoDB"
echo "[!] - Target location: $TARGET_LOC"

# Lista de IDs (Copiada do seu entrypoint)
ID_LIST=(8156 2335 9922 3039 4031 8386 7951 3538 9160 5746 1642 6139 2361 3456 9019 661 2818 7800 7536 4767 8565 4373 7901 7719 9278)

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
    # shellcheck disable=SC2093
    exec python migrate_data_mongo.py --loc $TARGET_LOC --filter_bs "$CLIENT_ID" --data_path "dataset/pecanstreet/15min/${TARGET_LOC}/train/" --test_path "dataset/pecanstreet/15min/${TARGET_LOC}/test/"\


    # Verifica se o python rodou com sucesso (Exit Code 0)
    if [ $? -eq 0 ]; then
        echo "[*] - Finished."
    else
        echo " [!] - Fail to migrate client $CLIENT_ID. Starting the next..."
    fi

    echo "------------------------------------------------"
done

echo "[!] - Migration finished!"