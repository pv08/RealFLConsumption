# RealFLConsumption

Framework de **Aprendizado Federado (FL)** para previsão de consumo de energia sobre o
**dataset Pecan Street** (medidores inteligentes, resolução de 15 min) em quatro localidades:
`austin`, `california`, `newyork`, `puertorico` (~23–25 clientes cada).

A camada de comunicação é um **protocolo binário próprio sobre sockets TCP** (`socket` +
`selectors`) com **long-polling** — não há gRPC/Flower/PySyft. O servidor é single-threaded
e todo o estado da simulação vive em `FLServerState` (`src/fl_manager.py`), que avança pelas
fases `WAITING_CLIENTS → INITIAL_EVAL → TRAINING → GLOBAL_EVAL → …`.

> Para detalhes de arquitetura interna e comportamentos não óbvios, veja o
> [`CLAUDE.md`](./CLAUDE.md).

---

## Sumário

1. [Instalação](#1-instalação)
2. [Variáveis de ambiente / `.env`](#2-variáveis-de-ambiente--env)
3. [Preparação dos dados (obrigatório)](#3-preparação-dos-dados-obrigatório)
4. [Execução local (servidor + clientes)](#4-execução-local-servidor--clientes)
5. [Cluster Docker Compose (`generate_simulation.py`)](#5-cluster-docker-compose-generate_simulationpy)
6. [Batch de vários modelos (`run_all_models.sh`)](#6-batch-de-vários-modelos-run_all_modelssh)
7. [Logging com Weights & Biases](#7-logging-com-weights--biases)
8. [Notificações via webhook](#8-notificações-via-webhook)
9. [Otimização de hiperparâmetros (Optuna)](#9-otimização-de-hiperparâmetros-optuna)
10. [Estrutura de saída](#10-estrutura-de-saída)
11. [Referência de features](#11-referência-de-features-estratégias-agregações-modelos)

---

## 1. Instalação

```bash
pip install -r requirements.txt
```

Dependências principais: `torch==2.9.1`, `numpy`, `pandas`, `scikit-learn`, `optuna`,
`optuna-dashboard`, `wandb`, `matplotlib`, `tqdm`, `pymongo`. Para GPU/determinismo, veja as
variáveis de ambiente abaixo.

> **Não há suíte de testes, linter ou build.** A validação é feita rodando uma simulação
> ponta a ponta.

---

## 2. Variáveis de ambiente / `.env`

Todas são **opt-in**: nada é enviado/logado a menos que a flag correspondente seja passada.
As chaves ficam em `.env` (comentadas por padrão) ou exportadas no shell.

| Chave | Usada por | Descrição |
|---|---|---|
| `WANDB_API_KEY` | `--enable_wandb` | Chave da conta Weights & Biases. **Obrigatória** para o log W&B. |
| `WANDB_PROJECT` | `--wandb_project` | Projeto W&B (default do arg: `fl_default`). |
| `WANDB_GROUP` | `--wandb_group` | Grupo W&B (default do arg: `default_group`). |
| `NOTIFY_WEBHOOK_URL` | `--enable_notifications` / `--notify` | Webhook Slack/Discord para notificar início/fim/erro. |
| `PYTORCH_ALLOC_CONF=expandable_segments:True` | GPU | Estabilidade de memória CUDA. |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | GPU | **Obrigatória** para determinismo em CUDA (`torch.use_deterministic_algorithms(True)`). |
| `OMP_NUM_THREADS`, `MALLOC_ARENA_MAX` | GPU | Knobs de throughput/memória. |

> Rodando em GPU local (fora do Docker), exporte `CUBLAS_WORKSPACE_CONFIG=:4096:8` ou a
> simulação pode falhar por causa do modo determinístico.
>
> Os scripts `run_all_models.sh` fazem `source .env` automaticamente. Ao rodar
> `docker compose up` na mão, exporte as chaves no shell antes (o compose lê `${VAR}` do
> shell que o invoca, não de dentro do container).

---

## 3. Preparação dos dados (obrigatório)

Converte os CSVs do Pecan Street em tensores NumPy. **Precisa ser feito antes de qualquer
treino.** Gera, por cliente, `{cid}-{train,val,test}-{X,y}.npy` + `{cid}_metadata.pkl`
(scalers e dims).

### Um cliente — `migrate_data_numpy.py`

```bash
python migrate_data_numpy.py --loc austin --filter_bs 661 \
  --data_path "dataset/pecanstreet/15min/austin/train/" \
  --test_path "dataset/pecanstreet/15min/austin/test/"
```

| Flag | Default | Descrição |
|---|---|---|
| `--loc` | `austin` | Localidade: `austin`, `california`, `newyork`, `puertorico`. |
| `--filter_bs` | `661` | ID do cliente (building/site). |
| `--data_path` | `dataset/pecanstreet/15min/austin/train/` | CSVs de treino. |
| `--test_path` | `dataset/pecanstreet/15min/austin/test/` | CSVs de teste. |
| `--test_size` | `0.2` | Fração de validação. |
| `--targets` | `['consumption']` | Alvo(s) da previsão. |
| `--num_lags` | `96` | Janela de defasagem (96 = 24 h a cada 15 min). |
| `--x_scaler` / `--y_scaler` | `minmax` | Normalização para `[-1, 1]`. |
| `--nan_constant` | `0` | Valor de imputação de NaN. |
| `--identifier` | `cid` | Coluna identificadora. |
| `--mongo_uri` | `mongodb://localhost:27017` | (Opcional) URI Mongo. |
| `--model_name` | `lstm` | Informativo nesta etapa. |

### Todos os clientes de uma localidade — `migrate_to_numpy.sh`

```bash
bash migrate_to_numpy.sh --loc austin      # austin | california | newyork | puertorico
```

Itera sobre a lista fixa de IDs da localidade e chama `migrate_data_numpy.py` para cada um.

---

## 4. Execução local (servidor + clientes)

Rode **um processo de servidor** e **um processo por cliente** (cada cliente em um terminal,
mudando `--filter_bs`). O servidor encerra sozinho quando a simulação termina **e** todos os
clientes obrigatórios devolvem seus testes finais.

### Terminal 1 — servidor (`app-server.py`)

```bash
python app-server.py \
  --required_clients 5 \
  --clients_per_round 3 \
  --max_rounds 10 \
  --client_strategy random \
  --aggregation fedavg
  # opcionais: --optimize_clients --enable_wandb --enable_notifications --disable_blockchain
```

| Flag | Default | Descrição |
|---|---|---|
| `--host` | `127.0.0.1` | Endereço de escuta. |
| `--port` | `65432` | Porta TCP. |
| `--required_clients` | `5` | Nº de clientes que precisam registrar para iniciar. |
| `--clients_per_round` | `5` | Quantos a estratégia seleciona por rodada. |
| `--max_rounds` | `2` | Nº de rodadas de FL. |
| `--client_strategy` | `random` | `random`, `fixed-representativeness`, `weekly-representativeness`. |
| `--min_cluster_size` | `2` | Tamanho mínimo de cluster (estratégias TimeVAE). |
| `--aggregation` | `fedavg` | Ver [tabela de agregações](#agregações---aggregation). |
| `--optimize_clients` | *(off)* | Ativa busca de hparams por cliente (Optuna). |
| `--disable_blockchain` | *(off)* | Pula o ledger Blockchain (sem hash/checagem de replay). |
| `--seed` | `0` | Semente global (reprodutibilidade). |
| `--epochs` | `None` | Informativo no servidor (o valor real é client-side). |
| `--loc` | `None` | Informativo (notificações/nome do run W&B). |
| `--enable_wandb` | *(off)* | Loga a simulação no W&B (run único no servidor). Requer `WANDB_API_KEY`. |
| `--enable_notifications` | *(off)* | Envia webhook em início/rodada/fim/erro. |
| `--wandb_project` | `$WANDB_PROJECT` ou `fl_default` | Projeto W&B. |
| `--wandb_group` | `$WANDB_GROUP` ou `default_group` | Grupo W&B. |

### Terminal 2..N — clientes (`app-client.py`)

```bash
python app-client.py \
  --filter_bs 661 \
  --model_name lstm \
  --loc austin \
  --epochs 10 \
  --data_path "dataset/pecanstreet/15min/austin/train/" \
  --test_path "dataset/pecanstreet/15min/austin/test/"
```

| Flag | Default | Descrição |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `65432` | Servidor a conectar. |
| `--filter_bs` | `0` | ID do cliente (deve ter dados migrados). |
| `--model_name` | `lstm` | `rnn`, `lstm`, `gru`, `cnn`. |
| `--loc` | `austin` | Localidade. |
| `--data_path` / `--test_path` | `.../austin/train/` `.../austin/test/` | Tensores migrados. |
| `--epochs` | `1` | Épocas de treino local por rodada. |
| `--lr` | `0.001` | Learning rate. |
| `--optimizer` | `adamw` | Otimizador. |
| `--batch_size` | `8` | Tamanho do batch. |
| `--criterion` | `mse` | Loss: `mse` ou `l1`. |
| `--num_lags` | `96` | Janela de defasagem. |
| `--early_stopping` | `False` | Early stopping. |
| `--patience` | `50` | Paciência do early stopping. |
| `--max_grad_norm` | `0.0` | Clip de gradiente (0 = desligado). |
| `--fedprox_mu` | `0.0` | Termo proximal do FedProx (aplicado client-side). |
| `--reg1` / `--reg2` | `0.0` | Regularização L1 / L2. |
| `--gpu_slots` | `1` | Slots de GPU (mutex por file-lock entre clientes co-locados). |
| `--seed` | `0` | Semente. |
| `--latent_dim` | `8` | Dimensão latente do TimeVAE (só p/ estratégias de representatividade). |
| `--timevae_epochs` | `1` | Épocas de treino do TimeVAE. |
| `--num_workers` | `0` | Workers do DataLoader. |

> `--required_clients` **gate** o início; `--clients_per_round` é quantos a estratégia
> escolhe por rodada.

---

## 5. Cluster Docker Compose (`generate_simulation.py`)

Gera um `docker-compose.gpu.<model>.<loc>.seed<seed>.yml` com o servidor + **um serviço GPU
por cliente** (lê a lista de IDs de `get_available_clients_location()` e ajusta
`--required_clients` para o total).

```bash
python generate_simulation.py --loc austin --model_name lstm \
  --max_rounds 10 --epochs 200 --clients_per_round 5
  # opcionais: --optimize_clients --enable_wandb --enable_notifications --disable_blockchain

docker compose -f docker-compose.gpu.lstm.austin.seed0.yml up --build
```

| Flag | Default | Descrição |
|---|---|---|
| `--loc` | `austin` | Localidade. |
| `--model_name` | `rnn` | `rnn`, `lstm`, `gru`, `cnn`. |
| `--host` / `--port` | `0.0.0.0` / `65432` | Bind do servidor no compose. |
| `--max_rounds` | `10` | Rodadas de FL. |
| `--epochs` | `200` | Épocas locais por cliente. |
| `--batch_size` | `1024` | Batch dos clientes. |
| `--clients_per_round` | `5` | Seleção por rodada. |
| `--num_workers` | `0` | Workers do DataLoader. |
| `--gpu_slots` | `1` | Slots de GPU por cliente. |
| `--seed` | `0` | Semente. |
| `--optimize_clients` | *(off)* | Optuna por cliente. |
| `--disable_blockchain` | *(off)* | Desliga o ledger. |
| `--enable_wandb` | *(off)* | Injeta `--enable_wandb` no serviço **servidor** (as envs `WANDB_*` só ficam no servidor). |
| `--enable_notifications` | *(off)* | Webhook no serviço servidor. |

> As variáveis `WANDB_API_KEY`/`WANDB_PROJECT`/`WANDB_GROUP` são substituídas via `${...}` a
> partir do shell que roda `docker compose up`. Exporte-as antes (ou use `run_all_models.sh`,
> que faz `source .env`).

---

## 6. Batch de vários modelos (`run_all_models.sh`)

Roda, para uma localidade, a sequência `rnn` (seed 0), `lstm` (seed 1), `gru` (seed 2):
gera o compose, sobe com `docker compose up`, aguarda o servidor terminar e derruba a stack.
Faz `source .env` automaticamente.

```bash
bash run_all_models.sh -loc austin
# com extras:
bash run_all_models.sh -loc austin -epochs 200 -clients_per_round 5 -gpu_slots 1 -notify -wandb
```

| Flag | Default | Descrição |
|---|---|---|
| `-loc` / `--loc` | *(obrigatório)* | `austin`, `california`, `newyork`, `puertorico`. |
| `-epochs` / `--epochs` | `200` | Épocas locais. |
| `-clients_per_round` / `--clients_per_round` | `5` | Seleção por rodada. |
| `-gpu_slots` / `--gpu_slots` | `1` | Slots de GPU por cliente. |
| `-notify` / `--notify` | *(off)* | Repassa `--enable_notifications` ao gerador. |
| `-wandb` / `--wandb` | *(off)* | Repassa `--enable_wandb` ao gerador. |

---

## 7. Logging com Weights & Biases

O log W&B é **opt-in** e centralizado no **servidor** (um único run por simulação). Nada é
logado sem a flag, e nenhuma chamada é feita se `WANDB_API_KEY` não estiver no ambiente.

**Ativar:**

```bash
export WANDB_API_KEY=<sua_chave>
python app-server.py ... --enable_wandb
# ou no cluster: python generate_simulation.py ... --enable_wandb
# ou no batch:   bash run_all_models.sh -loc austin -wandb
```

**O que é logado (1 run, eixo-x = `round`):**

| Namespace | Métricas |
|---|---|
| `server/` | `global_loss`, `best_loss`, `train_loss`, `val_loss` + métricas globais ponderadas por rodada. |
| `client/{cid}/` | `train_loss`, `val_loss`, `time_spent` por cliente selecionado. |
| `selection/client_{cid}` | `0`/`1` indicando quem foi sorteado na rodada. |
| `config` | strategy, aggregation, clients_per_round, required_clients, max_rounds, seed, loc, epochs, optimize_clients. |

O nome do run é `"{loc}-{aggregation}-seed{seed}"`; projeto/grupo vêm de
`--wandb_project`/`--wandb_group` (ou das envs). O run é encerrado com `wandb.finish()` em
qualquer caminho de saída (fim normal, Ctrl-C ou erro).

---

## 8. Notificações via webhook

Opt-in via `--enable_notifications` (server/gerador) ou `-notify` (batch). Requer
`NOTIFY_WEBHOOK_URL` (Slack/Discord). Envia mensagem em início, cada rodada, fim, interrupção
manual e crash. Implementação em `src/utils/notifier.py`.

---

## 9. Otimização de hiperparâmetros (Optuna)

Com `--optimize_clients`, o servidor mantém um **estudo Optuna por cliente**
(`study_<cid>` em `optuna_db/fl_simulation_<Model>.db`), otimizando `lr`, `batch_size`,
`optimizer` (+ `fedprox_mu` quando a agregação é `fedprox`) contra a **loss de validação**.

Dashboard:

```bash
optuna-dashboard sqlite:///optuna_db/fl_simulation_<ModelName>.db
```

---

## 10. Estrutura de saída

Tudo sob `etc/` (gitignored):

| Caminho | Conteúdo |
|---|---|
| `etc/fl/server/ckpt/<Model>/` | Melhores checkpoints do modelo global (`.pth`). |
| `etc/fl/local/ckpt/<Model>/<cid>/` | Checkpoints locais por cliente. |
| `etc/fl/logs/<Model>/` | `history_simulation.pkl`, losses por rodada, `blockchain_ledger.jsonl`. |
| `etc/fl/results/<Model>/` | `global_model_cids_tests.pkl` (testes finais). |
| `etc/TimeVAE/<loc>/` | Checkpoints/logs do TimeVAE (cache entre execuções). |
| `optuna_db/` | Estudos SQLite do Optuna. |

---

## 11. Referência de features (estratégias, agregações, modelos)

### Estratégias de seleção de clientes (`--client_strategy`)

| Valor | Comportamento |
|---|---|
| `random` | Amostragem uniforme por rodada (reprodutível por seed). |
| `fixed-representativeness` | Clusteriza por assinatura latente (TimeVAE) **uma vez** e fixa o comitê. |
| `weekly-representativeness` | Re-clusteriza a cada `rounds_per_week` rodadas. ⚠️ *não wired end-to-end — cai em random* (ver `CLAUDE.md`). |

### Agregações (`--aggregation`)

| Valor | Tipo | Observação |
|---|---|---|
| `fedavg` | Ponderada | FedAvg padrão. |
| `avg` | Simples | Média não ponderada. |
| `medianavg` | Mediana | Robusta a outliers. |
| `fedprox` | Ponderada | Agrega como FedAvg; termo proximal `mu` é **client-side** (`--fedprox_mu`). |
| `fedadagrad` / `fedyogi` / `fedadam` / `fedavgm` | **Stateful** | Persistem momentum/variância entre rodadas. |
| `fednova` | — | ⚠️ **quebrado via CLI** (mismatch de chave — ver `CLAUDE.md`). |

### Modelos de previsão (`--model_name`)

`rnn`, `lstm`, `gru`, `cnn` — entrada `(batch, lags, input_dim)`, saída `output_dim` passos.
`input_dim`/`output_dim` vêm do metadata de cada cliente (definidos na preparação dos dados).
O **TimeVAE** é usado **apenas** para gerar assinaturas latentes de clustering — nunca como
modelo de previsão.

> **Limitações conhecidas** (`fednova` quebrado, `weekly-representativeness` incompleto,
> `RoundRobinSelection` não plugado, etc.) estão documentadas em detalhe no
> [`CLAUDE.md`](./CLAUDE.md).
