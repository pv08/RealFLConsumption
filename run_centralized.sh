#!/usr/bin/env bash
# Treino centralizado em lote: percorre (regiao x modelo x entidade) chamando
# train_centralized.py, com um pool de execucoes paralelas.
#
# A ordem da lista de jobs e regiao -> modelo -> cid, de modo que -jobs N
# coloque N entidades do MESMO modelo e da MESMA regiao rodando juntas
# (ex.: -scope clients -jobs 5 = 5 clientes ao mesmo tempo para aquele modelo;
#  -scope community -jobs 3 = 3 modelos ao mesmo tempo para aquela regiao,
#  ja que a comunidade tem uma unica entidade por regiao).
#
# Nao usa src/utils/gpu_lock.py de proposito: aquele mutex serializa por design
# (e aponta para /app/lock_dir, caminho de Docker). Aqui o paralelismo e o
# objetivo e o throttle e a flag -jobs.
#
# Uso tipico:
#   bash run_centralized.sh -loc austin -models "lstm" -epochs 10 -jobs 5
#   bash run_centralized.sh -loc all -bootstrap -jobs 5 -notify

# Sem -e: um job que falha nao pode derrubar o lote (mesma politica de
# run_all_models.sh). O pipefail vale para os pipes de contagem do sumario.
set -uo pipefail

# Este script so funciona a partir da raiz do repositorio: os caminhos abaixo
# sao relativos e cids_of() importa `src.utils.functions` via `python -c`, que
# resolve pelo diretorio corrente. Ancorar aqui deixa o runner chamavel de
# qualquer lugar.
cd "$(dirname "$(readlink -f "$0")")" || { echo "[!] - Nao consegui entrar na raiz do repo."; exit 1; }

PYTHON="${PYTHON:-/home/paulo/anaconda3/envs/FLConsumption/bin/python}"
DATA_ROOT="${DATA_ROOT:-dataset/pecanstreet/15min}"

TARGET_LOC="all"
MODELS="rnn lstm gru"
SCOPE="both"
JOBS=3
NO_RETRY=false
EPOCHS=200
BATCH_SIZE=1024
LR=0.001
OPTIMIZER=adamw
SEED=0
OUT_ROOT="etc/in"
BOOTSTRAP=false
FORCE=false
DRY_RUN=false
NOTIFY=false

usage() {
    cat <<'USAGE'
Uso: bash run_centralized.sh [opcoes]

  -loc         austin|california|newyork|puertorico|all   (default: all)
  -models      lista entre aspas                          (default: "rnn lstm gru")
  -scope       clients|community|both                     (default: both)
  -jobs        tamanho do pool paralelo                   (default: 3)
               ~2,5 GiB de VRAM por job com batch 1024; ~1,3 GiB com 512.
               Jobs que estourarem a memoria sao repetidos em serie no fim.
  -epochs      epocas por execucao                        (default: 200)
  -batch_size  batch de treino                            (default: 1024)
  -lr          taxa de aprendizado                        (default: 0.001)
  -optimizer   otimizador                                 (default: adamw)
  -seed        semente                                    (default: 0)
  -out_root    raiz dos artefatos                         (default: etc/in)
  -bootstrap   gera community.csv (se faltar) e migra os .npy que faltarem
  -force       reexecuta jobs cujos artefatos ja existem  (default: pula)
  -dry-run     imprime a lista de jobs e sai
  -no_retry    nao repete em serie os jobs que falharem por CUDA OOM
  -notify      envia o sumario final pelo NOTIFY_WEBHOOK_URL do .env
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --loc|-loc)                TARGET_LOC="$2"; shift 2 ;;
        --models|-models)          MODELS="$2"; shift 2 ;;
        --scope|-scope)            SCOPE="$2"; shift 2 ;;
        --jobs|-jobs)              JOBS="$2"; shift 2 ;;
        --epochs|-epochs)          EPOCHS="$2"; shift 2 ;;
        --batch_size|-batch_size)  BATCH_SIZE="$2"; shift 2 ;;
        --lr|-lr)                  LR="$2"; shift 2 ;;
        --optimizer|-optimizer)    OPTIMIZER="$2"; shift 2 ;;
        --seed|-seed)              SEED="$2"; shift 2 ;;
        --out_root|-out_root)      OUT_ROOT="$2"; shift 2 ;;
        --bootstrap|-bootstrap)    BOOTSTRAP=true; shift ;;
        --force|-force)            FORCE=true; shift ;;
        --dry-run|-dry-run)        DRY_RUN=true; shift ;;
        --no_retry|-no_retry)      NO_RETRY=true; shift ;;
        --notify|-notify)          NOTIFY=true; shift ;;
        --help|-h)                 usage; exit 0 ;;
        *) echo "[!] - Argumento desconhecido: $1"; usage; exit 1 ;;
    esac
done

# Notificacoes sao opt-in (-notify); NOTIFY_WEBHOOK_URL vem do .env.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

notify() {
    if [[ "$NOTIFY" == "true" ]]; then
        "$PYTHON" -m src.utils.notifier "$1" || true
    fi
}

case "$SCOPE" in
    clients|community|both) ;;
    *) echo "[!] - -scope invalido: '$SCOPE'. Use clients, community ou both."; exit 1 ;;
esac

ALL_LOCS=(austin california newyork puertorico)
case "$TARGET_LOC" in
    all) LOCS=("${ALL_LOCS[@]}") ;;
    austin|california|newyork|puertorico) LOCS=("$TARGET_LOC") ;;
    *) echo "[!] - -loc invalido: '$TARGET_LOC'. Use austin, california, newyork, puertorico ou all."; exit 1 ;;
esac

for M in $MODELS; do
    case "$M" in
        rnn|lstm|gru|cnn) ;;
        *) echo "[!] - Modelo invalido: '$M'. Use rnn, lstm, gru ou cnn."; exit 1 ;;
    esac
done

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
    echo "[!] - -jobs precisa ser um inteiro >= 1 (recebido: '$JOBS')."; exit 1
fi

# Unica fonte de verdade para a lista de clientes: a mesma funcao que o servidor
# FL e o generate_simulation.py usam. Evita o array recopiado de migrate_to_numpy.sh.
cids_of() {
    local out rc errfile
    errfile=$(mktemp)
    # stdout e stderr SEPARADOS. Juntar os dois com 2>&1 faz qualquer aviso de
    # import (matplotlib sem HOME gravavel, por exemplo) virar "cliente" na
    # lista, e o runner sai procurando um <aviso>.csv que nao existe.
    out=$("$PYTHON" -c "from src.utils.functions import get_available_clients_location as g; print(' '.join(map(str, g('$1'))))" 2>"$errfile")
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[!] - Nao consegui listar os clientes de '$1' com $PYTHON:" >&2
        sed 's/^/        /' "$errfile" >&2
        echo "        Verifique se o ambiente tem as dependencias (requirements.txt)." >&2
        rm -f "$errfile"
        exit 1
    fi
    rm -f "$errfile"
    if [[ -z "${out// }" ]]; then
        echo "[!] - Lista de clientes vazia para '$1'." >&2
        exit 1
    fi
    # Rede de seguranca: cid e sempre numerico. Se algo nao-numerico passar,
    # e ruido, nao cliente.
    local c
    for c in $out; do
        if ! [[ "$c" =~ ^[0-9]+$ ]]; then
            echo "[!] - Lista de clientes de '$1' contem entrada invalida: '$c'" >&2
            echo "        Saida completa: $out" >&2
            exit 1
        fi
    done
    echo "$out"
}

# Resolvido uma vez so: `VAR="$(cids_of ...)"` propaga o status do subshell,
# entao um import quebrado aborta aqui em vez de virar lista vazia la na frente.
declare -A CIDS_OF_LOC
for LOC in "${LOCS[@]}"; do
    CIDS_OF_LOC[$LOC]="$(cids_of "$LOC")" || exit 1
done

entities_of() {
    local loc="$1" out=()
    [[ "$SCOPE" == "clients" || "$SCOPE" == "both" ]] && out+=(${CIDS_OF_LOC[$loc]})
    [[ "$SCOPE" == "community" || "$SCOPE" == "both" ]] && out+=("community")
    printf '%s\n' "${out[@]}"
}

# Um job esta pronto quando os 8 artefatos existem E foram gerados com o mesmo
# numero de epocas pedido agora. So a existencia nao basta: um teste rapido de
# -epochs 5 deixa os 8 arquivos no lugar, e sem esta checagem uma corrida
# posterior com -epochs 200 pularia tudo e entregaria modelos de 5 epocas.
# A contagem sai do cabecalho do .npy, que traz a forma em ASCII nos ~128
# primeiros bytes. (Com -early_stopping a curva pode ser mais curta de forma
# legitima; nesse caso o job e refeito, que e o lado seguro do erro.)
job_done() {
    local loc="$1" m="$2" cid="$3" base="$OUT_ROOT/$1"
    local curve="$base/logs/$m/${cid}_val_mse.npy"
    [[ -f "$curve" ]] || return 1
    head -c 128 "$curve" 2>/dev/null | tr -d '\0' | grep -q "'shape': ($EPOCHS,)" || return 1
    [[ -f "$base/results/$m/${cid}_results.json" ]] &&
    [[ -f "$base/results/$m/${cid}_results_inverted.json" ]] &&
    [[ -f "$base/results/$m/preds/${cid}_preds.json" ]] &&
    [[ -f "$base/results/$m/preds/inverted/${cid}_preds.json" ]] &&
    [[ -f "$base/logs/$m/${cid}_train_mse.npy" ]] &&
    [[ -f "$base/logs/$m/${cid}_val_mse.npy" ]] &&
    [[ -f "$base/ckpts/$m/best/${m}_${cid}.pth" ]] &&
    [[ -f "$base/ckpts/$m/best/${m}_${cid}.h5" ]]
}

RUNLOG_DIR="$OUT_ROOT/_runlogs"
STAMP="$(date +%Y%m%d-%H%M%S)"
STATUS_FILE="$RUNLOG_DIR/status-$STAMP.tsv"
# Falhar aqui, e alto. Em Docker, um bind mount cujo caminho no host nao existe
# e criado como root:root, e o container - que roda com o uid do usuario - nao
# escreve nele. Sem esta checagem o lote seguia adiante e terminava com
# "OK: 0 | FAIL: 0" e codigo de saida 0, ou seja, parecia sucesso.
if ! mkdir -p "$RUNLOG_DIR" 2>/dev/null || ! : > "$STATUS_FILE" 2>/dev/null; then
    echo "[!] - Sem permissao de escrita em '$OUT_ROOT' (uid $(id -u), gid $(id -g))." >&2
    echo "        Dono do diretorio: $(stat -c '%U:%G' "$OUT_ROOT" 2>/dev/null || echo 'nao existe')" >&2
    echo "        Em Docker, isso costuma ser um volume criado como root porque o" >&2
    echo "        caminho nao existia no host. Crie-o antes: mkdir -p $OUT_ROOT" >&2
    exit 1
fi

echo "[!] - loc=$TARGET_LOC | models=\"$MODELS\" | scope=$SCOPE | jobs=$JOBS"
echo "[!] - epochs=$EPOCHS | batch_size=$BATCH_SIZE | lr=$LR | optimizer=$OPTIMIZER | seed=$SEED"
echo "[!] - out_root=$OUT_ROOT | bootstrap=$BOOTSTRAP | force=$FORCE"

# ---------------------------------------------------------------------------
# Etapa 1 - bootstrap dos dados: community.csv e os .npy que faltarem.
#
# Nada disso vem do git (/dataset/ e /etc/ estao no .gitignore). O que precisa
# existir de antemao sao os CSV por cliente <loc>/{train,test}/<cid>.csv; a
# serie comunitaria e os .npy sao derivados deles e reproduziveis em qualquer
# maquina. Custo: ~1 GB de .npy por entidade.
# ---------------------------------------------------------------------------
MISSING_CSV=()
MISSING_NPY=()
for LOC in "${LOCS[@]}"; do
    while read -r CID; do
        [[ -z "$CID" ]] && continue
        if [[ "$CID" == "community" ]]; then
            if [[ ! -f "$DATA_ROOT/$LOC/train/community.csv" ]]; then
                if [[ "$BOOTSTRAP" == "true" ]]; then
                    echo "[*] - Gerando a serie comunitaria de $LOC..."
                    [[ "$DRY_RUN" == "true" ]] ||
                        "$PYTHON" tools/build_community_dataset.py --loc "$LOC" --root "$DATA_ROOT" ||
                        { echo "[!] - Falha ao gerar community.csv de $LOC"; exit 1; }
                else
                    MISSING_CSV+=("$LOC/community.csv (rode com -bootstrap)")
                fi
            fi
        elif [[ ! -f "$DATA_ROOT/$LOC/train/$CID.csv" ]]; then
            MISSING_CSV+=("$LOC/$CID.csv")
        fi
        [[ -f "$DATA_ROOT/$LOC/train/$CID-train-X.npy" ]] || MISSING_NPY+=("$LOC	$CID")
    done < <(entities_of "$LOC")
done

if [[ ${#MISSING_CSV[@]} -gt 0 ]]; then
    echo "[!] - Faltam CSV de entrada (nao sao derivaveis; precisam ser copiados):"
    printf '        %s\n' "${MISSING_CSV[@]}"
    exit 1
fi

if [[ ${#MISSING_NPY[@]} -gt 0 ]]; then
    if [[ "$BOOTSTRAP" != "true" ]]; then
        echo "[!] - Faltam .npy para ${#MISSING_NPY[@]} entidade(s). Rode com -bootstrap para migrar:"
        printf '        %s\n' "${MISSING_NPY[@]}"
        exit 1
    fi
    echo "[!] - Migrando ${#MISSING_NPY[@]} entidade(s) para .npy (~1 GB cada)..."
    if [[ "$DRY_RUN" == "true" ]]; then
        printf '        [migrate] %s\n' "${MISSING_NPY[@]}"
    else
        MIGFILE="$(mktemp)"
        printf '%s\n' "${MISSING_NPY[@]}" > "$MIGFILE"
        # Migracao e CPU/pandas e escreve arquivos disjuntos por cid: paraleliza
        # sem risco de corrida, e roda ANTES do treino para nao disputar a GPU.
        export PYTHON DATA_ROOT RUNLOG_DIR
        migrate_one() {
            local loc="$1" cid="$2"
            mkdir -p "$RUNLOG_DIR/$loc"
            echo "    [migrate] $loc/$cid"
            "$PYTHON" migrate_data_numpy.py --loc "$loc" --filter_bs "$cid" \
                --data_path "$DATA_ROOT/$loc/train/" \
                --test_path "$DATA_ROOT/$loc/test/" \
                > "$RUNLOG_DIR/$loc/migrate_$cid.log" 2>&1 ||
                echo "    [migrate] FALHOU $loc/$cid -> $RUNLOG_DIR/$loc/migrate_$cid.log"
        }
        export -f migrate_one
        xargs -a "$MIGFILE" -P "$JOBS" -L1 bash -c 'migrate_one "$@"' _
        rm -f "$MIGFILE"

        STILL=()
        for entry in "${MISSING_NPY[@]}"; do
            IFS=$'\t' read -r loc cid <<< "$entry"
            [[ -f "$DATA_ROOT/$loc/train/$cid-train-X.npy" ]] || STILL+=("$loc/$cid")
        done
        if [[ ${#STILL[@]} -gt 0 ]]; then
            echo "[!] - Migracao falhou para: ${STILL[*]}"
            exit 1
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Etapa 2 - lista de jobs, na ordem regiao -> modelo -> cid.
# ---------------------------------------------------------------------------
JOBFILE="$(mktemp)"
trap 'rm -f "$JOBFILE"' EXIT

N_TOTAL=0
N_SKIP=0
for LOC in "${LOCS[@]}"; do
    for M in $MODELS; do
        while read -r CID; do
            [[ -z "$CID" ]] && continue
            N_TOTAL=$((N_TOTAL + 1))
            if [[ "$FORCE" != "true" ]] && job_done "$LOC" "$M" "$CID"; then
                N_SKIP=$((N_SKIP + 1))
                printf '%s\t%s\t%s\tSKIP\n' "$LOC" "$M" "$CID" >> "$STATUS_FILE"
                continue
            fi
            printf '%s\t%s\t%s\n' "$LOC" "$M" "$CID" >> "$JOBFILE"
        done < <(entities_of "$LOC")
    done
done

N_QUEUED=$(wc -l < "$JOBFILE" | tr -d ' ')
echo "[!] - $N_TOTAL job(s) no total | $N_SKIP ja prontos (pulados) | $N_QUEUED a executar"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "------------------------------------------------"
    cat "$JOBFILE"
    echo "------------------------------------------------"
    echo "[!] - dry-run: nada foi executado."
    exit 0
fi

if [[ "$N_QUEUED" -eq 0 ]]; then
    echo "[!] - Nada a fazer."
    exit 0
fi

# ---------------------------------------------------------------------------
# Etapa 3 - pool paralelo.
# ---------------------------------------------------------------------------
export PYTHON OUT_ROOT RUNLOG_DIR STATUS_FILE DATA_ROOT
export EPOCHS BATCH_SIZE LR OPTIMIZER SEED N_QUEUED

# seed_all chama torch.use_deterministic_algorithms(True), que em CUDA exige a
# variavel abaixo. Cada job e um processo separado que semeia a si mesmo, entao
# a ordem de escalonamento do pool nao afeta o resultado de nenhum deles.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Sem isto, N processos torch simultaneos abrem cada um um pool do tamanho da
# maquina e o paralelismo vira prejuizo.
export OMP_NUM_THREADS=1
# Mesmo knob que generate_simulation.py poe nos containers: com varios processos
# dividindo a mesma GPU, a fragmentacao do alocador e o que estoura primeiro.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Aviso de orcamento de VRAM. Medido nesta base: ~2,5 GiB por processo com
# batch 1024 e ~1,3 GiB com 512, para lstm/gru/rnn com hidden 128 e 96 lags.
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ -n "${VRAM_MIB:-}" ]]; then
    PER_JOB_MIB=$(( BATCH_SIZE >= 1024 ? 2600 : (BATCH_SIZE >= 512 ? 1400 : 900) ))
    FITS=$(( VRAM_MIB / PER_JOB_MIB ))
    [[ "$FITS" -lt 1 ]] && FITS=1
    if [[ "$JOBS" -gt "$FITS" ]]; then
        echo "[!] - Aviso: -jobs $JOBS com batch $BATCH_SIZE estima ~$((JOBS * PER_JOB_MIB)) MiB"
        echo "        contra ${VRAM_MIB} MiB de VRAM. Cabem ~$FITS. Os jobs que"
        echo "        estourarem serao repetidos em serie no fim (ou use -jobs $FITS)."
    fi
fi

run_one() {
    local loc="$1" m="$2" cid="$3"
    local logdir="$RUNLOG_DIR/$loc/$m"
    mkdir -p "$logdir"
    local logfile="$logdir/$cid.log"
    local t0=$SECONDS

    # Trava por job. Sem ela, dois runners lancados por engano treinam o mesmo
    # cliente ao mesmo tempo e gravam por cima um do outro. mkdir e atomico.
    # Um lock cujo PID morreu (runner interrompido) e tomado, para nao travar
    # o job para sempre.
    local lockdir="$RUNLOG_DIR/.locks/${loc}_${m}_${cid}"
    mkdir -p "$RUNLOG_DIR/.locks"
    if ! mkdir "$lockdir" 2>/dev/null; then
        local owner
        owner=$(cat "$lockdir/pid" 2>/dev/null)
        if [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null; then
            echo "    [lock] $loc/$m/$cid ja em execucao (pid $owner); pulando"
            return 0
        fi
        echo "    [lock] $loc/$m/$cid tinha lock orfao; assumindo"
        rm -rf "$lockdir" && mkdir "$lockdir" 2>/dev/null || return 0
    fi
    echo $$ > "$lockdir/pid"

    if "$PYTHON" train_centralized.py \
            --loc "$loc" \
            --model_name "$m" \
            --filter_bs "$cid" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --optimizer "$OPTIMIZER" \
            --criterion mse \
            --num_lags 96 \
            --seed "$SEED" \
            --out_root "$OUT_ROOT" \
            --data_path "$DATA_ROOT/$loc/train/" \
            --test_path "$DATA_ROOT/$loc/test/" \
            > "$logfile" 2>&1; then
        printf '%s\t%s\t%s\tOK\n' "$loc" "$m" "$cid" >> "$STATUS_FILE"
        echo "    [ok]   $loc/$m/$cid  ($((SECONDS - t0))s)"
    else
        printf '%s\t%s\t%s\tFAIL\n' "$loc" "$m" "$cid" >> "$STATUS_FILE"
        echo "    [FAIL] $loc/$m/$cid  -> $logfile"
    fi
    rm -rf "$lockdir"
}
export -f run_one

START_TS=$SECONDS
echo "------------------------------------------------"
xargs -a "$JOBFILE" -P "$JOBS" -L1 bash -c 'run_one "$@"' _

# Um job que estoura a VRAM nao falhou por defeito, falhou por disputa: com a
# GPU livre ele passa. Repete em serie so esses, uma vez.
if [[ "$NO_RETRY" != "true" ]]; then
    RETRYFILE="$(mktemp)"
    while IFS=$'\t' read -r loc m cid st; do
        [[ "$st" == "FAIL" ]] || continue
        # Falta de VRAM aparece com mais de uma assinatura: alem do
        # torch.OutOfMemoryError, cuBLAS falha ao alocar o proprio handle
        # (CUBLAS_STATUS_ALLOC_FAILED) quando a GPU ja esta cheia.
        grep -qE "OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: out of memory|CUDA out of memory" \
            "$RUNLOG_DIR/$loc/$m/$cid.log" 2>/dev/null &&
            printf '%s\t%s\t%s\n' "$loc" "$m" "$cid" >> "$RETRYFILE"
    done < "$STATUS_FILE"
    N_RETRY=$(wc -l < "$RETRYFILE" | tr -d ' ')
    if [[ "$N_RETRY" -gt 0 ]]; then
        echo "------------------------------------------------"
        echo "[!] - $N_RETRY job(s) estouraram a VRAM. Repetindo em serie..."
        xargs -a "$RETRYFILE" -P 1 -L1 bash -c 'run_one "$@"' _
    fi
    rm -f "$RETRYFILE"
fi

ELAPSED=$((SECONDS - START_TS))
echo "------------------------------------------------"

# ---------------------------------------------------------------------------
# Etapa 4 - sumario.
# ---------------------------------------------------------------------------
FINAL_FILE="$RUNLOG_DIR/final-$STAMP.tsv"
awk -F'\t' '{ last[$1"\t"$2"\t"$3] = $4 } END { for (k in last) print k "\t" last[k] }' \
    "$STATUS_FILE" | sort > "$FINAL_FILE"

n_ok=$(awk -F'\t' '$4=="OK"' "$FINAL_FILE" | wc -l | tr -d ' ')
n_fail=$(awk -F'\t' '$4=="FAIL"' "$FINAL_FILE" | wc -l | tr -d ' ')
n_skip=$(awk -F'\t' '$4=="SKIP"' "$FINAL_FILE" | wc -l | tr -d ' ')

echo "[!] - OK: $n_ok | FAIL: $n_fail | SKIP: $n_skip | tempo: $((ELAPSED / 60))min"
echo "[!] - Status: $FINAL_FILE (bruto, com retentativas: $STATUS_FILE)"

if [[ "$n_fail" -gt 0 ]]; then
    echo "[!] - Jobs que falharam:"
    awk -F'\t' -v d="$RUNLOG_DIR" '$4=="FAIL" {printf "        %s/%s/%s -> %s/%s/%s/%s.log\n", $1,$2,$3, d,$1,$2,$3}' "$FINAL_FILE"
fi

SUMMARY="Treino centralizado (loc=$TARGET_LOC, models=\"$MODELS\", scope=$SCOPE): OK $n_ok | FAIL $n_fail | SKIP $n_skip em $((ELAPSED / 60))min"
notify "$SUMMARY"

[[ "$n_fail" -eq 0 ]]
