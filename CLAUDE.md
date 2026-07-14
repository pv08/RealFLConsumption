# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RealFLConsumption is a federated learning (FL) simulation framework for energy consumption forecasting. It uses the **Pecan Street dataset** (15-minute resolution smart meter data) from four locations: `austin`, `california`, `newyork`, `puertorico` (~23–25 clients each). The FL system is built on **raw Python TCP sockets with a long-polling protocol** — there is no gRPC/Flower/PySyft dependency; the coordination logic is hand-rolled.

The user refers to the transport as "websocket" communication, but the actual implementation is a custom binary protocol over `socket` + `selectors` (see Communication Layer below).

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Data preparation (CSV → NumPy, required before any training)
```bash
# One client
python migrate_data_numpy.py --loc austin --filter_bs 661 \
  --data_path "dataset/pecanstreet/15min/austin/train/" \
  --test_path "dataset/pecanstreet/15min/austin/test/"

# All clients for a location (valid: austin | newyork | california | puertorico)
bash migrate_to_numpy.sh --loc austin
```
This produces per-client `{cid}-{train,val,test}-{X,y}.npy` files plus `{cid}_metadata.pkl` (holds pickled `x_scaler`/`y_scaler` and `input_dim`/`output_dim`).

### Run a simulation locally (server + one process per client)
```bash
# Terminal 1 — server
python app-server.py \
  --required_clients 5 \
  --clients_per_round 3 \
  --max_rounds 10 \
  --client_strategy random \      # or: fixed-representativeness | weekly-representativeness
  --aggregation fedavg            # or: avg | medianavg | fedprox | fedadagrad | fedyogi | fedadam | fedavgm
  # add --optimize_clients to enable per-client Optuna hyperparameter search

# Terminal 2..N — one client per process (change --filter_bs each time)
python app-client.py \
  --filter_bs 661 \
  --model_name lstm \             # or: rnn | gru | cnn
  --loc austin \
  --epochs 10 \
  --data_path "dataset/pecanstreet/15min/austin/train/" \
  --test_path "dataset/pecanstreet/15min/austin/test/"
```
`required_clients` gates the start of training; `clients_per_round` is how many the selection strategy picks per round. The server process exits by itself once the simulation is over **and** every required client has returned its final test results.

### Generate a Docker Compose cluster (server + all clients as GPU services)
```bash
python generate_simulation.py --loc austin --model_name lstm \
  --max_rounds 10 --epochs 200 --clients_per_round 5 [--optimize_clients]
# writes docker-compose.gpu.<model>.<loc>.yml

docker compose -f docker-compose.gpu.lstm.austin.yml up --build
```
`generate_simulation.py` reads the client id list per location from `get_available_clients_location()` in `src/utils/functions.py` and sets `--required_clients` to the full list size.

### Optuna dashboard (only relevant with --optimize_clients)
```bash
optuna-dashboard sqlite:///optuna_db/fl_simulation_<ModelName>.db
```

There is **no test suite, linter, or build step** in this repo — validation is done by running a simulation end-to-end.

## Architecture

### Communication Layer (`src/comm/`)

Server (`libserver.Message`) and clients (`libclient.Message`) talk over raw TCP using a custom framing: `ProtoHeader (2-byte big-endian length) + pickled JSON-header + pickled payload`. **Everything is serialized with `pickle`** (numpy arrays, torch state, dicts all flow directly).

**Long-polling is the core coordination primitive.** When a client checks in and the server has no task ready (waiting for more clients, or a round hasn't started), the server returns `task == "defer"` and **holds the TCP connection open without replying**. Connections are parked in `FLServerState.pending_messages`. When state changes, `_notify_pending_clients()` / `_notify_all_stop()` re-evaluate each parked connection and push the response via `trigger_delayed_response()`. There are **no threads and no asyncio** — a single-threaded `selectors.DefaultSelector` loop (`app-server.py:main`) drives non-blocking I/O for every connection.

The client side (`app-client.py:send_and_wait`) is fully **synchronous**: it opens a fresh connection per request and blocks its own selector loop until a response arrives.

### FL Orchestration (`src/fl_manager.py`)

`FLServerState` is the single shared state machine. All connections mutate it. It advances through four phases:

```
WAITING_CLIENTS → INITIAL_EVAL → TRAINING → GLOBAL_EVAL → (loop to TRAINING, or stop)
```

- **WAITING_CLIENTS** — defers everyone until `required_clients` have registered; then builds the global model architecture from the first client's reported dims.
- **INITIAL_EVAL** — every client evaluates the randomly-initialized global model. For `TimeVAE`-based selection, clients additionally return a latent-space signature here.
- **TRAINING** — the selection strategy picks `clients_per_round`; selected clients get global weights (+ Optuna hparams if enabled) and train locally; non-selected clients are deferred.
- **GLOBAL_EVAL** — all clients evaluate the freshly aggregated model; then either a new round starts or `max_rounds` is hit and `stop` is broadcast.

The `check_task(client_id, message_obj)` method is the router that maps `(phase, client_id)` → one of `"defer" | "evaluate" | "train" | "stop"`. Every client update is passed through the **Blockchain ledger** (`src/structure/blockchain.py`): a SHA-256 of the pickled weights is checked against seen hashes to reject duplicate/replay contributions before aggregation.

### Client Learning (`src/client_learning.py`, `src/utils/process_executor.py`)

`ClientLearning` owns local training, evaluation, final test, and TimeVAE latent extraction. Training and evaluation are launched in **isolated subprocesses** via `ProcessExecutor` (`torch.multiprocessing` with the `spawn` context) so CUDA memory is fully reclaimed between rounds. Results are returned through a `mp.Queue`.

**GPU slot arbitration** (`src/utils/gpu_lock.py`): a file-lock mutex (`fcntl.flock` on `<lock_dir>/gpu_{i}.lock`) serializes CUDA work across co-located client processes. Slot count = `--gpu_slots`. The default `lock_dir` is `/app/lock_dir` (a Docker path) — running locally outside Docker requires that path to be writable or the default changed.

### Client Selection Strategies (`src/base/selection_strategy.py`)

| CLI value | Class | Behavior |
|---|---|---|
| `random` | `RandomSelection` | Uniform random sample each round |
| `fixed-representativeness` | `TimeVAE` | Cluster clients by latent signature **once**, cache the committee permanently |
| `weekly-representativeness` | `TimeVAEWeeklyRepresentativeSelection` | Re-cluster every `rounds_per_week` rounds |

TimeVAE strategies sweep 8 distance metrics (euclidean, squared-euclidean, manhattan, cosine, hassanat, minkowski, chebyshev, canberra) and pick the one with the best **bootstrap centroid stability** (`BaseClustering.centroid_stability` in `src/base/clustering.py`), then take the medoid of each cluster as its representative. `BaseClustering` wraps agglomerative clustering on a precomputed distance matrix and enforces `min_cluster_size` by stealing members from larger clusters.

### Aggregation (`src/base/aggregation_strategy.py`, `src/utils/aggregation_functions.py`)

`Aggregator.aggregate(weights_list, current_model)` dispatches on `self.alg`. Weighted/simple/median variants are stateless; `fedadagrad`, `fedyogi`, `fedadam`, `fedavgm` are **stateful** — they persist momentum/variance vectors (`m_t`, `v_t`, `momentum_vector`) on the `Aggregator` instance across rounds. `fedprox` uses plain FedAvg aggregation on the server; its proximal `mu` term is applied **client-side** in `ClientLearning.train`.

### Forecasting Models (`src/models/`)

`RNN`, `LSTM`, `GRU`, `CNN` all take `(batch, lags, input_dim)` and output `output_dim` steps; built via `get_model()` in `src/utils/functions.py` (hidden size 128, 1 layer by default, `matrix_rep=True`). `input_dim`/`output_dim` come from each client's metadata, computed at preprocessing time.

**TimeVAE** (`src/models/timeVAE/`) is a variational autoencoder for time series used **only** to produce latent client signatures for clustering — it is never the forecasting model. Trained TimeVAE checkpoints are cached at `etc/TimeVAE/<loc>/ckpt/{cid}-latent_dim_{d}.pth` and reused across runs.

### Dataset Pipeline (`src/dataset/`)

`participant_preprocessing.py` (`ParticipantData`) turns raw Pecan Street CSVs into feature frames (grid→`consumption`, solar→`generation`, temporal fields, weather join via `dataset/pecanstreet/weather_data/open-meteo-<loc>.csv`). `processing.py` (`Processing`, subclass of `Data`) handles NaN imputation, train/val split, scaling (default `minmax` to `[-1, 1]`), time-lag generation (default `num_lags=96` = 24h at 15-min sampling), and reshaping to `(samples, lags, features)`. The single default target is `consumption`. `LocalFileDataset` (`src/data.py`) just loads the resulting `.npy` tensors.

### Output Structure (all under `etc/`, gitignored)
- `etc/fl/server/ckpt/<Model>/` — best global checkpoints (`.pth`, keyed by loss/round)
- `etc/fl/local/ckpt/<Model>/<cid>/` — per-client local checkpoints
- `etc/fl/logs/<Model>/` — `history_simulation.pkl`, per-round local loss `.npy`, `blockchain_ledger.json`
- `etc/fl/results/<Model>/` — final `global_model_cids_tests.pkl`
- `etc/TimeVAE/<loc>/` — cached TimeVAE checkpoints + logs
- `optuna_db/` — per-model SQLite Optuna studies

### Hyperparameter Optimization

With `--optimize_clients`, the server keeps a **per-client Optuna study** (`study_<cid>` in `optuna_db/fl_simulation_<Model>.db`). Before each training assignment it `ask()`s for `lr`, `batch_size`, `optimizer` (+ `fedprox_mu` when aggregation is `fedprox`), ships them to the client, and `tell()`s the study the returned **validation loss** after the update arrives.

## Non-obvious behaviors & gotchas

These are things that will bite during new implementations — verify against the code before relying on them:

- **`fednova` is broken via CLI.** `Aggregator.aggregate()` matches the key `"fednova_aggregate"`, but `get_params()`/`__repr__()` use `"fednova"`. Passing `--aggregation fednova` silently returns empty weights; passing `fednova_aggregate` raises `NotImplementedError` in `__repr__`. Fix the key mismatch before using FedNova.
- **Aggregator hyperparameters are initialized as a side effect of `__repr__()`.** `self.mu/rho/beta_1/eta/tau/...` are only set inside `Aggregator.__repr__`. It works today only because `FLServerState.__init__` logs `repr(self.aggr_strategy)` at startup. Do **not** remove that log call, or move the assignments into `__init__`, or stateful aggregators will `AttributeError` in `aggregate()`.
- **`weekly-representativeness` is not wired end-to-end.** `TimeVAEWeeklyRepresentativeSelection.select()` indexes `client_data['latent_space'][ "week_N" ]`, but `ClientLearning.get_latent_space()` returns a single flat vector (not a per-week dict). As written it will warn and fall back to random. Needs the client to emit week-keyed signatures.
- **`RoundRobinSelection` exists but is unreachable.** It's defined in `selection_strategy.py` but not wired into `get_select_strategy()` in `app-server.py`; only `random`/`fixed-representativeness`/`weekly-representativeness` are selectable.
- **Determinism requires a CUBLAS env var on GPU.** `seed_all()` calls `torch.use_deterministic_algorithms(True)`. On CUDA this needs `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set only in the *client* Docker env in `generate_simulation.py`). Local GPU runs may crash without exporting it.
- **WandB logging is entirely commented out.** Every `wandb.init/log` block in `fl_manager.py` and `client_learning.py` is disabled, even though `WANDB_*` args/env are still wired. Re-enabling means uncommenting those blocks, not just setting env vars.
- **The evaluate path computes the latent space twice.** In `app-client.py`, `ProcessExecutor.run_evaluate` computes the latent space in a subprocess, then the result is discarded (`latent_space = None`) and recomputed in the main process via `trainer.get_latent_space(...)`. Functional but redundant.
- **`.env` is committed to git** (despite `.gitignore` listing `*.env`, because it was already tracked) and currently contains a live `WANDB_API_KEY`. Treat that key as compromised; do not add new secrets to a tracked file.

## Environment Variables

`.env` / Docker environment:
- `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_GROUP` — WandB integration (wired but currently disabled in code)
- `CUDA_LAUNCH_BLOCKING`, `PYTORCH_ALLOC_CONF=expandable_segments:True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS`, `MALLOC_ARENA_MAX` — GPU stability/determinism knobs set in the generated Docker Compose
