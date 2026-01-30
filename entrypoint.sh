#!/bin/bash

# Define a localidade padrão se não for passada via ENV
TARGET_LOC=${LOCALITY:-"austin"}

echo "--- Setting FL Clients ---"
echo "Target Location: $TARGET_LOC"

# 1. Definição das Listas de IDs por Localidade
# IMPORTANTE: Use parênteses ( ) para definir arrays no Bash
case $TARGET_LOC in
  "austin")
    # Sua lista completa da primeira localidade
    ID_LIST=(8156 2335 9922 3039 4031 8386 7951 3538 9160 5746 1642 6139 2361 3456 9019 661 2818 7800 7536 4767 8565 4373 7901 7719 9278)
    ;;

  *)
    echo "[!]Error: '$TARGET_LOC' location not found on entrypoint.sh"
    exit 1
    ;;
esac

# Verifica se carregou
NUM_IDS=${#ID_LIST[@]}
echo "Total IDs available in this location: $NUM_IDS"

# 2. Descobre o Índice deste Container (0, 1, 2...)
# O hostname geralmente é "projeto_servico_1", pegamos o número no final
HOSTNAME_INDEX=$(hostname | grep -o '[0-9]*$' | tail -1)

if [ -z "$HOSTNAME_INDEX" ]; then
    # Fallback: Se não tiver número no hostname, gera um hash aleatório
    # Útil para debug ou execução fora do docker compose scale
    echo "[!] Warning: It was not possible extract hostname indices. Using Hash."
    checksum=$(hostname | cksum | cut -f1 -d" ")
    MY_INDEX=$(( checksum % NUM_IDS ))
else
    # O Docker inicia a contagem em 1, mas arrays em bash começam em 0
    # Então subtraímos 1.
    MY_INDEX=$((HOSTNAME_INDEX - 1))
fi

# 3. Garante que o índice esteja dentro dos limites (Round Robin)
# Se você subir 30 containers para uma lista de 25 IDs, o 26º container pega o 1º ID
SAFE_INDEX=$(( MY_INDEX % NUM_IDS ))

# 4. A MÁGICA: Pega o VALOR REAL baseado no índice
REAL_CLIENT_ID=${ID_LIST[$SAFE_INDEX]}

echo "------------------------------------------------"
echo "Container Hostname : $(hostname)"
echo "Indices Computed   : $SAFE_INDEX"
echo "ID REAL (filter_bs): $REAL_CLIENT_ID"
echo "------------------------------------------------"

# 5. Executa a aplicação passando o ID REAL
# O exec substitui o processo shell pelo python (economiza memória)
exec python app-client.py \
    --host "${FL_SERVER_HOST:-fl-server}" \
    --port "${FL_SERVER_PORT:-65432}" \
    --data_path "dataset/pecanstreet/15min/${TARGET_LOC}/train/"\
    --test_path "dataset/pecanstreet/15min/${TARGET_LOC}/test/"\
    --filter_bs "$REAL_CLIENT_ID" \
    --epochs "${EPOCHS:-5}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --num_workers 0