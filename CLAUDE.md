# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RealFLConsumption is a federated learning (FL) simulation framework for energy consumption forecasting in the energy sector. It uses the **Pecan Street dataset** (15-minute resolution smart meter data) from four locations: Austin, California, New York, and Puerto Rico. The FL system is built on raw Python TCP sockets with a **long-polling protocol** — no gRPC/Flower/PySyft dependency.

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Data preparation (CSV → NumPy format, required before training)
```bash
# Migrate one client
python migrate_data_numpy.py --loc austin --filter_bs 661 \
  --data_path "dataset/pecanstreet/15min/austin/train/" \
  --test_path "dataset/pecanstreet/15min/austin/test/"

# Migrate all clients for a location
bash migrate_to_numpy.sh --loc austin  # options: austin, ny, california, puerto_rico
```

### Run a simulation locally (two terminals)
```bash
# Terminal 1 — server
python app-server.py \
  --required_clients 5 \
  --clients_per_round 3 \
  --max_rounds 10 \
  --client_strategy random   # or: fixed-representativeness, weekly-representativeness
  --aggregation fedavg       # or: avg, medianavg, fedprox, fedadagrad, fedyogi, fedadam, fedavgm

# Terminal 2 — one client (repeat per client, changing --filter_bs)
python app-client.py \
  --filter_bs 661 \
  --model_name lstm \         # or: rnn, gru, cnn
  --loc austin \
  --epochs 10 \
  --data_path "dataset/pecanstreet/15min/austin/train/" \
  --test_path "dataset/pecanstreet/15min/austin/test/"
```

### Generate Docker Compose for GPU cluster
```bash
python generate_simulation.py \
  --loc austin \
  --model_name lstm \
  --max_rounds 10 \
  --epochs 200 \
  --clients_per_round 5 \
  [--optimize_clients]
# Outputs: docker-compose.gpu.lstm.austin.yml

docker compose -f docker-compose.gpu.lstm.austin.yml up --build
```

### Optuna dashboard (hyperparameter tuning results)
```bash
optuna-dashboard sqlite:///optuna_db/fl_simulation_<ModelName>.db
```

## Architecture

### Communication Layer (`src/comm/`)

The server and clients communicate over raw TCP sockets using a custom binary protocol: `ProtoHeader (2 bytes) + PickledHeader + PickledPayload`. All messages are serialized with `pickle`.

**Long-polling pattern**: when a client checks in and there is no task ready (e.g., waiting for more clients to register, or waiting for a training round to start), the server **holds the TCP connection open** without responding (`task == "defer"`). When the state changes, `FLServerState._notify_pending_clients()` wakes all deferred connections and sends their tasks. This is the central coordination mechanism — no background threads or async I/O.

The server uses `selectors.DefaultSelector` (non-blocking I/O, single thread). Each client connection is a `libserver.Message` object registered with the selector. Clients (`libclient.Message`) use the same selector approach.

### FL Orchestration (`src/fl_manager.py`)

`FLServerState` is the single state machine shared across all connections. It progresses through four phases:

```
WAITING_CLIENTS → INITIAL_EVAL → TRAINING → GLOBAL_EVAL → (loop or stop)
```

- **WAITING_CLIENTS**: defers all clients until `required_clients` have checked in.
- **INITIAL_EVAL**: all clients evaluate the randomly-initialized global model. For `TimeVAE`-based selection strategies, clients also compute and return their latent space signatures.
- **TRAINING**: selected clients receive global weights and train locally. Non-selected clients are deferred.
- **GLOBAL_EVAL**: all clients evaluate the new aggregated global model. After evaluation, either starts a new round or signals `stop`.

Each client update is validated through a **Blockchain ledger** (`src/structure/blockchain.py`) using SHA-256 hashes of model weights to detect duplicate/replay contributions.

### Client Learning (`src/client_learning.py`, `src/utils/process_executor.py`)

`ClientLearning` handles local training and TimeVAE-based latent space extraction. Training and evaluation run in **isolated subprocesses** via `ProcessExecutor` (using `torch.multiprocessing` with `spawn` context) to ensure CUDA memory is fully released between rounds.

**GPU slot management** (`src/utils/gpu_lock.py`): file-lock-based mutex (`fcntl.flock`) on `lock_dir/gpu_{i}.lock` files prevents multiple clients on the same machine from running CUDA kernels simultaneously. Controlled by `--gpu_slots`.

### Client Selection Strategies (`src/base/selection_strategy.py`)

| Strategy | Class | Behavior |
|---|---|---|
| `random` | `RandomSelection` | Random sampling each round |
| `fixed-representativeness` | `TimeVAE` | Clusters clients by latent space once (INITIAL_EVAL), caches the committee permanently |
| `weekly-representativeness` | `TimeVAEWeeklyRepresentativeSelection` | Re-clusters every `rounds_per_week` rounds |

TimeVAE selection evaluates 8 distance metrics (euclidean, manhattan, cosine, hassanat, etc.) via bootstrap centroid stability to pick the best clustering metric, then selects one representative per cluster.

### Aggregation Algorithms (`src/base/aggregation_strategy.py`, `src/utils/aggregation_functions.py`)

`Aggregator` dispatches to: `fedavg`, `avg` (simple), `medianavg`, `fedprox` (uses FedAvg aggregation, mu applied client-side), `fednova`, `fedadagrad`, `fedyogi`, `fedadam`, `fedavgm`. Stateful aggregators (`fedadagrad`, `fedyogi`, `fedadam`, `fedavgm`) store momentum/variance vectors on the `Aggregator` instance between rounds.

### Forecasting Models (`src/models/`)

All models (RNN, LSTM, GRU, CNN) accept `(batch, lags, input_dim)` tensors and predict `output_dim` steps. Default architecture: hidden size 128, 1 layer. Instantiated via `get_model()` in `src/utils/functions.py`.

**TimeVAE** (`src/models/timeVAE/`) is a VAE for time series used exclusively to generate latent representations for client clustering. It is not the forecasting model.

### Dataset Pipeline

Raw CSV files from Pecan Street are preprocessed by `src/dataset/participant_preprocessing.py` (feature engineering: lag features, temporal features, weather join) and `src/dataset/processing.py` (train/val/test split, scaling). The output is per-client `.npy` files plus a `{cid}_metadata.pkl` with scalers and dimensions. `LocalFileDataset` (`src/data.py`) loads these `.npy` files directly into PyTorch tensors.

### Output Structure

All simulation artifacts land under `etc/`:
- `etc/fl/server/ckpt/<Model>/` — best global model checkpoints (`.pth`)
- `etc/fl/local/ckpt/<Model>/<cid>/` — per-client local model checkpoints
- `etc/fl/logs/<Model>/` — `history_simulation.pkl`, per-round local loss `.npy` files, blockchain ledger JSON
- `etc/fl/results/<Model>/` — final test results (`global_model_cids_tests.pkl`)
- `etc/TimeVAE/<loc>/ckpt/` — cached TimeVAE models per client
- `optuna_db/` — SQLite databases for Optuna per-model hyperparameter studies

### Hyperparameter Optimization

When `--optimize_clients` is passed to the server, each client gets hyperparameters (`lr`, `batch_size`, `optimizer`, and `fedprox_mu` for FedProx) suggested by a per-client Optuna study. The server tells the trial result after receiving the client's validation loss.

### Environment Variables

`.env` / Docker environment:
- `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_GROUP` — WandB integration (currently commented out in code but wired in Docker Compose)
- `PYTORCH_ALLOC_CONF=expandable_segments:True` — GPU stability setting used in Docker (`CUDA_LAUNCH_BLOCKING=1` is intentionally not set by default — it forces synchronous kernel launches and was killing training throughput; set it manually only when debugging an async CUDA error)
