#!/usr/bin/env bash
# Treina o modelo comunitario centralizado para cada regiao e arquitetura.
#
# Wrapper fino sobre run_centralized.sh, preservado por procedencia: fixa os
# parametros exatos que geraram os resultados comunitarios ja verificados
# (batch 512, do Anexo B; rnn/lstm/gru da Tabela 8 + cnn por paridade de
# artefatos). O default de run_centralized.sh e batch 1024 - o mesmo que
# generate_simulation.py passa aos clientes federados -, entao chamar o runner
# direto SUBSTITUIRIA os artefatos comunitarios por versoes com outro batch.
#
# Pre-requisito (feito automaticamente por -bootstrap):
#   python tools/build_community_dataset.py --loc all
#   python migrate_data_numpy.py --loc <loc> --filter_bs community ...
#
# Uso:
#   bash run_community.sh                      # 4 regioes x 4 modelos
#   bash run_community.sh -loc austin -jobs 3  # 3 modelos da mesma regiao juntos
exec bash "$(dirname "$(readlink -f "$0")")/run_centralized.sh" \
    -scope community \
    -models "rnn lstm gru cnn" \
    -batch_size 512 \
    "$@"
