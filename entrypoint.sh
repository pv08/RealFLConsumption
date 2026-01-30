#!/bin/bash

ID_LIST=($CLIENT_ID_LIST)

INDEX=$(hostname | grep -o '[0-9]*$' | tail -1)

if [ -z "$INDEX" ]; then
    # Gera um número entre 0 e o tamanho da lista - 1
    SEED=$(hostname | cksum | cut -f1 -d" ")
    INDEX=$(( SEED % ${#ID_LIST[@]} ))
else
    # Se o índice existir (ex: replica 1), ajustamos para base 0
    INDEX=$((INDEX - 1))
fi

CLIENT_ID=${ID_LIST[$INDEX]}

echo "Container Hostname: $(hostname)"
echo "Client identification: $CLIENT_ID"

# Executa o cliente
python app-client.py \
    --host "${FL_SERVER_HOST:-fl-server}" \
    --port "${FL_SERVER_PORT:-65432}" \
    --filter_bs "$CLIENT_ID" \
    --epochs "${EPOCHS:-200}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --num_workers 0