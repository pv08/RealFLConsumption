#!/usr/bin/env bash
# Mostra o andamento de um lote de run_centralized.sh.
#
# O runner redireciona cada job para seu proprio arquivo em
# <out_root>/_runlogs/<loc>/<modelo>/<cid>.log, porque N treinos em paralelo
# escrevendo no mesmo terminal ficam ilegiveis. Este script le esses arquivos.
#
# Uso:
#   bash progress.sh                 # uma foto do estado atual
#   bash progress.sh -w              # atualiza a cada 10s
#   bash progress.sh -out_root etc/in.smoke
cd "$(dirname "$(readlink -f "$0")")" || exit 1

OUT_ROOT="etc/in"
WATCH=false
INTERVAL=10
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out_root|-out_root) OUT_ROOT="$2"; shift 2 ;;
        --watch|-w|-watch)    WATCH=true; shift ;;
        --interval|-interval) INTERVAL="$2"; shift 2 ;;
        *) echo "[!] - Argumento desconhecido: $1"; exit 1 ;;
    esac
done

RUNLOG_DIR="$OUT_ROOT/_runlogs"

snapshot() {
    local status_file
    status_file=$(ls -t "$RUNLOG_DIR"/status-*.tsv 2>/dev/null | head -1)
    if [[ -z "$status_file" ]]; then
        echo "[!] - Nenhum lote encontrado em $RUNLOG_DIR"
        return 1
    fi

    echo "=== lote $(basename "$status_file" .tsv | sed 's/^status-//') | $(date +%H:%M:%S) ==="

    # Jobs vivos, com a epoca corrente. tqdm/log usam \r, dai o tr.
    local any=false
    while read -r pid args; do
        [[ -z "$pid" ]] && continue
        any=true
        local loc m cid
        loc=$(sed -E 's/.*--loc ([^ ]+).*/\1/' <<< "$args")
        m=$(sed -E 's/.*--model_name ([^ ]+).*/\1/' <<< "$args")
        cid=$(sed -E 's/.*--filter_bs ([^ ]+).*/\1/' <<< "$args")
        local log="$RUNLOG_DIR/$loc/$m/$cid.log"
        local ep="(iniciando)"
        [[ -f "$log" ]] && ep=$(tr '\r' '\n' < "$log" | grep -oE "Epoch [0-9]+/[0-9]+" | tail -1)
        printf "  rodando  %-12s %-4s %-10s %s\n" "$loc" "$m" "$cid" "${ep:-(carregando dados)}"
    done < <(ps -eo pid,args | grep "[t]rain_centralized.py")
    [[ "$any" == "false" ]] && echo "  (nenhum job em execucao)"

    # Contagem final por job: o status e append-only e uma retentativa
    # acrescenta uma segunda linha, entao vale a ultima de cada (loc,m,cid).
    awk -F'\t' '{ last[$1"\t"$2"\t"$3] = $4 } END {
        for (k in last) c[last[k]]++
        printf "  concluidos: OK %d | FAIL %d | SKIP %d\n", c["OK"], c["FAIL"], c["SKIP"]
    }' "$status_file"

    awk -F'\t' '{ last[$1"\t"$2"\t"$3] = $4 } END {
        for (k in last) if (last[k] == "FAIL") { split(k, p, "\t"); print "  FALHOU: " p[1] "/" p[2] "/" p[3] }
    }' "$status_file"
}

if [[ "$WATCH" == "true" ]]; then
    while true; do
        clear
        snapshot
        sleep "$INTERVAL"
    done
else
    snapshot
fi
