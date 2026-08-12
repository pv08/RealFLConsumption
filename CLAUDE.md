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

The `check_task(client_id, message_obj)` method is the router that maps `(phase, client_id)` → one of `"defer" | "evaluate" | "train" | "stop"`. Every client update is passed through the **Blockchain ledger** (`src/structure/blockchain.py`): a SHA-256 hashed directly over each weight array's raw bytes (no pickling) is checked against seen hashes to reject duplicate/replay contributions before aggregation. The ledger itself is appended to incrementally as JSON Lines, not rewritten in full on every block.

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

#### TimeVAE selection — end-to-end flow

Both TimeVAE strategies (`fixed-representativeness`, `weekly-representativeness`) hinge on a **per-client latent signature** produced once, during `INITIAL_EVAL`:

1. **Server → client.** During `INITIAL_EVAL`, `check_task` sets `req_latent_space=True` (because the strategy class name contains `"TimeVAE"`) and `latent_mode` (`"weekly"` iff the strategy is `TimeVAEWeeklyRepresentativeSelection`, else `"fixed"`) in the `evaluate` payload.
2. **Client trains/loads a per-client TimeVAE.** `ClientLearning.get_latent_space(latent_dim, timevae_epochs, mode, samples_per_week)` (run inside the isolated evaluate subprocess, under `GPULock`) looks for a cached checkpoint at `etc/TimeVAE/<loc>/ckpt/<cid>-latent_dim_<d>.pth`. If missing it trains a TimeVAE (`train_timevae`, a proper `for epoch in range(timevae_epochs)` loop, AdamW + `ReduceLROnPlateau`, best-by-val-loss) and saves it; otherwise it loads it. The forecasting model is never used here — TimeVAE is only for signatures.
3. **Client emits the signature** from the encoder's `z_mean` over its (chronologically-ordered) training windows:
   - `mode="fixed"` → a single flat mean vector.
   - `mode="weekly"` → a dict `{f"week_{i}": mean_vector}`, slicing the ordered windows into consecutive blocks of `samples_per_week` (default 672 = 7 days × 96 steps/day at 15 min).
4. **Server clusters and selects.** `receive_metrics` pops `latent_space` into `registered_clients[cid]`. `_start_training_phase` calls `selection_strategy.select(registered_clients, clients_per_round)`, which builds `X_latents`, sweeps the 8 metrics for best centroid stability, clusters into `clients_per_round` clusters, and returns one **medoid** per cluster. Selected clients get `train`; the rest stay long-polling.
   - `fixed` caches the committee **permanently** (computed once, round 0).
   - `weekly` recomputes per window: `week_idx = ((round_counter-1)//rounds_per_week)+1`, clamped to the max week index actually present in the clients' dicts (so late rounds saturate on the last week instead of falling back to random), and re-clusters when the window changes.

**Relevant flags.** Server: `--client_strategy {fixed,weekly}-representativeness`, `--min_cluster_size`, `--rounds_per_week` (weekly only). Client: `--latent_dim`, `--timevae_epochs`, `--hidden_dims`, `--samples_per_week` (weekly only), `--reconstruction_wt`, `--trend_poly`, `--use_residual_conn`.

**Caveats / recommended values.**
- The client-side defaults are intentionally tiny for smoke tests (`--timevae_epochs 1`, `--hidden_dims [2,4,8]`, `--latent_dim 8`); for meaningful signatures pass e.g. `--timevae_epochs 50 --hidden_dims 128,256,512`. (Note `--hidden_dims` uses argparse `type=list`, so a CLI string is split per-character — set the default in code or pass a real list.) **The reference repo's own default is `[50,100,200]`, not `[128,256,512]`** — worth sweeping both.
- **All clients must use the same `--latent_dim`** — it is the clustering space dimension and must match across clients.
- The **selection** checkpoint filename encodes only `latent_dim`, **not** `hidden_dims`/`trend_poly`/etc. Changing the TimeVAE architecture while keeping the same `latent_dim` will make `load_state_dict(strict=True)` fail against a stale cached checkpoint — **clear `etc/TimeVAE/<loc>/ckpt/` when you change the architecture.** (The *generative* checkpoints no longer have this problem: their filename carries a config hash — see "Checkpoints" below.)
- Signatures are only collected at `INITIAL_EVAL`; weekly re-clustering reuses those once-computed per-week vectors (it does not re-extract latents mid-simulation).

#### TimeVAE generative evaluation (TSTR / TRTS) — `eval_timevae.py`

A **standalone, offline, per-client** script (`eval_timevae.py`, repo root) that assesses how good the TimeVAE is as a **generative** model, independent of the FL loop. Two frameworks:
- **TSTR** (Train-on-Synthetic, Test-on-Real): fit a generator on real train data, sample synthetic windows, train a forecaster on them, and evaluate on the **real test set**. Measures whether generated data can train a useful model.
- **TRTS** (Train-on-Real, Test-on-Synthetic): train a forecaster on real train data and evaluate it on **synthetic** windows (generator fit on real test data). Measures whether a real-trained model performs on generated data.

**Generative representation (replaces the reference repo's `concat_X_y` hack).** The reference appended the scalar target `y` as a constant 45th channel repeated across all lags — degenerate (the VAE sees a flat line) and redundant (`consumption` is already a channel of `X`). Instead, the generator here models a window of the **real** multivariate series of length `num_lags + output_dim` (97) over the `input_dim` (44) real channels. Synthetic `X`/`y` are then split **by time**: `X_syn = W[:, :num_lags]`, `y_syn = W[:, num_lags:, cons_idx]`. The 97-step windows are stitched from consecutive real windows (`W[i] = concat(X[i], X[i+1][-1:])`, exploiting stride-1 overlap) so the extra step carries **real** features — nothing fabricated. `cons_idx` (the consumption channel, index 40 for austin) is derived robustly by the exact match `X[i,-1,c] == y[i-1]`. All of this lives in `ClientLearning`: `_derive_consumption_index`, `_stitch_windows`, `build_generative_windows`, `train_generative_timevae`, `sample_synthetic`.

**Generation is by prior sampling** (`z ~ N(0, I) → decoder(z)`), not reconstruction — truly novel windows. The regressor reuses `ClientLearning` (`get_model`/`train`/`test`/`accumulate_metrics`/`inverse_transform_test`).

**Configurable VAE training.** `ClientLearning.train_timevae` takes `lr`, `beta` (β-VAE KL weight), `kl_anneal_epochs` (linearly ramp KL 0→beta to avoid prior collapse/spread), and `reconstruction_wt`. Defaults (`lr=1e-3, beta=1.0, kl_anneal_epochs=0, reconstruction_wt=None→args`) reproduce the selection-VAE behavior exactly. `best_model` is selected by validation loss evaluated at the **final** beta (fair across annealing epochs).

**Hyperparameter optimization (Optuna, per-client).** `ClientLearning.optimize_generative_timevae` runs a per-client study (`gen_study_<cid>_<split>` in `optuna_db/timevae_generation_<loc>.db`) searching `latent_dim ∈ {8,16,32,64}`, `lr`, `reconstruction_wt`, `beta`, and `kl_anneal_frac`. The **objective is a TSTR-style real-validation MSE**: each trial trains the generator (reduced epochs, no cache), prior-samples synthetic, trains a quick regressor on it, and evaluates on the **real val set** (`_tstr_val_score`) — this penalizes both poor reconstruction and a degenerate prior. The regressor's `best_model` is picked on an **inner split of the synthetic data**, so the real val set is used only for scoring (it used to do both, making the objective optimistic). The search space also covers `mean_time_wt`/`mean_feat_wt` separately and `cons_wt` (consumption channel weight). The best config is retrained for full epochs and cached with a `-opt` tag; best params are written to `etc/TimeVAE/<loc>/logs/<cid>-gen-<split>-opt-best_params.json`. Enable via `--optimize` in `eval_timevae.py`.

**Generator train/val split.** `_generative_loaders` shuffles before holding out 20% (seeded by `args.seed`), matching the reference's `split_data(shuffle=True)`. It used to take the chronological tail, so the generator never saw the most recent period and its val loss came from a different regime — which biased both `best_model` selection and the HPO. `--no_gen_val_shuffle` restores the old behavior.

**Diagnostic plots (`src/utils/graphs.py`).** `plot_baseline_vs_opt` (2×2: target curve real vs reconstructed with MSE/Pearson r, marginal distribution, mean ±1σ profile, individual windows) and `plot_loss_ablation` (1×2) compare two generators on the same client/split; both build on `reconstruct_windows` (deterministic, decodes `z_mean`), `prior_windows`, and `to_kw`. `plot_tsne` ports the reference's `visualize_and_save_tsne` — real windows vs prior samples in t-SNE space; unlike the reference it defaults to the **consumption channel only** (`channel=None` reproduces the reference's mean over all channels, which with 44 channels dilutes the one that matters). Output goes to `etc/TimeVAE/<loc>/results/plots/`.

**Checkpoints.** The generative TimeVAE is cached separately from the selection one, at `etc/TimeVAE/<loc>/ckpt/<cid>-gen-<split><tag>-ld<d>-<hash8>.pth` (`split` ∈ `train`/`test`; `tag` empty or `-opt`), with `seq_len = num_lags + output_dim` (97) vs the selection VAE's `seq_len = num_lags` (96) — they are **not** interchangeable. `<hash8>` is a SHA-256 prefix over everything that changes the resulting weights (latent_dim, hidden_dims, trend_poly, custom_seas, seq_len, channel weights, all loss weights, batch size, window options) — so two different loss shapes can no longer silently share a cache entry. The full config is mirrored to `etc/TimeVAE/<loc>/logs/<same-stem>-config.json`. Pre-hash checkpoints named `<cid>-gen-<split><tag>-latent_dim_<d>.pth` are still loaded as a **fallback** when no hashed file matches, with a loud log line saying the config is unverified. Metrics (scaled + inverse-transformed) are written to `etc/TimeVAE/<loc>/results/<Model>/<cid>_<TSTR|TRTS|BASELINE>_metrics_ld<d>.csv`.

**Channel weighting — the dominant knob.** The generator models all `input_dim` (44) channels and the reconstruction loss sums uniformly over `97 × 44 = 4268` elements, so **consumption gets ~1% of the objective** (measured: 1.11%, rank 25/44). The single most expensive channel is `minute` at **16.4%** — a deterministic 4-value sawtooth. Of the 44 channels, ~30 are weather and 8 are deterministic calendar fields. `--channel_wt "consumption=20,generation=5"` reweights the loss per channel (unnamed channels stay at 1.0). Measured at a matched 20-epoch budget: consumption MSE 0.02484 → 0.02000 (R² vs mean 0.727 → 0.780), generation 0.02459 → 0.01897, other channels 0.02213 → 0.02973, `minute` 0.129 → 0.362 — capacity moves off the channels nobody needs. **Note the share of the unweighted error is a misleading metric** (it falls when consumption improves and the rest degrades); compare absolute per-channel MSE.

Channel names come from `ClientLearning.channel_names()`: the window channel index maps to `x_scaler.feature_names_in_` **reversed** (`canal_j ↔ feature_names_in_[F-1-j]`), because `Processing.generate_time_lags` reverses columns before `to_timeseries_rep`. A constructor-time assertion cross-checks `channel_index("consumption")` against `_derive_consumption_index()` and fails loudly if the preprocessing order ever changes.

**Reconstruction loss terms.** `mean_axis_wt` used to weight **both** mean-matching terms; they are now independent — `--mean_feat_wt` (reduces `dim=2`, keeps the time axis; the term the reference **enables**) and `--mean_time_wt` (reduces `dim=1`; the term the reference leaves **commented out**, since a flat reconstruction at the right level satisfies it). Generative default is the reference's (`mean_time_wt=0, mean_feat_wt=1`); passing the legacy `--mean_axis_wt` applies one value to both. The **selection** VAE keeps `1.0/1.0` and is bit-for-bit unchanged (verified numerically, including the summation order).

**Trend/Seasonal are now real layers.** They used to be instantiated inside `TimeVAEDecoder.forward()` (a bug inherited from the reference): fresh random weights every call, never registered as parameters, never trained, never saved. With the defaults (`trend_poly=0`, `custom_seas=None`) it stayed latent and the model was effectively `level + residual conv` — a plain conv VAE, **not** TimeVAE. They are built in `__init__` now, so `--trend_poly N` and `--custom_seas "24x4"` actually work. Caveat: `SeasonalLayer` indexes by **position in the window**, so it only means "time of day" with `--align_to_day`.

**Window sampling.** `--window_stride` / `--align_to_day` control which stitched windows are kept. `--align_to_day` selects windows starting at 00:00 **by predicate** on the `hour`/`minute` channels, not by arithmetic stride — client 661's series has a discontinuity (one 92-step gap), so `W[start::96]` drifts out of phase (only 114/325 windows landed at midnight; by predicate it is 325/325). Cost: 31,248 → 325 training windows.

**Conditional generation (`--gen_channels`) — the current default.** The decoder models only the 6 endogenous meter channels (`consumption`, `generation`, `leg1v`, `leg2v`, `prev_consumption`, `consumption_change`) and is **conditioned** on the other 38 (30 weather + 8 calendar), which are exogenous or deterministic. `ConditionEncoder` (`inner_layers.py`) compresses the exogenous slice to a `--cond_dim` (32) vector; the decoder sub-layers are built with `latent_dim + cond_dim` and receive `cat([z, c])`, so **no sub-layer changed**. At sampling time the context comes from a real window (cycled through a seeded permutation so every context is used before any repeats) and the output is reassembled to the full 44 channels — everything downstream keeps the same shape. `--gen_channels` must include `consumption` (else `y_syn` would be copied from real data) and cannot cover all channels; both fail loudly. `TimeVAE(gen_idx=None)` is the unconditional model, byte-identical to before, and the **selection** VAE is untouched (`app-client.py` has its own parser and never sets these flags).

**Held-out evaluation (`--gen_holdout_days`).** Reserves the last N days of the train+val series and excludes them from `build_generative_windows`, **plus a `seq_len-1` window buffer** — without the buffer the last training window shares 96 of 97 steps with the first reserved one. This matters more than it sounds: evaluating on the generator's own random 80/20 split showed a memorisation gap of only +0.09/+0.17, while the contiguous block showed reconstruction degrading **10–17×** (r 0.96 → 0.60/0.66). Random splits over stride-1 windows cannot measure generalisation. Measured on the block, **resampling real training windows (W1 = 0.382) beats both generators (0.478 / 0.624)** — the generator does not beat the trivial baseline on an unseen period. Off by default (0), since training on everything is the operational config; turn it on to evaluate.

**Flags.** Core: `--loc --filter_bs --model_name --mode {TSTR,TRTS,both,baseline} --latent_dim --timevae_epochs --n_synthetic --hidden_dims --r_epochs --r_batch_size --batch_size`. Generator training: `--gen_lr --beta --kl_anneal_frac --channel_wt --mean_time_wt --mean_feat_wt --var_wt --grad_wt --trend_poly --custom_seas --window_stride --align_to_day --no_gen_val_shuffle`. Conditional/held-out: `--gen_channels --cond_dim --gen_holdout_days`. GPU arbitration: `--gpu_slots --lock_dir` (below).

**Running many clients in parallel (`--gpu_slots`).** `eval_timevae.py` is one client per invocation, so evaluating a whole location means many concurrent processes on one GPU. It reuses the FL simulation's mechanism: with `--gpu_slots N` (N ≥ 1) each **phase** — `optimize`, `TSTR`, `TRTS`, `baseline`, `plots` — runs in an isolated `spawn` subprocess (`ProcessExecutor.run_timevae_phase`) held by a `GPULock`, so only N runs touch the GPU at a time and CUDA memory is fully reclaimed between phases. `--gpu_slots 0` (**the default**) keeps the old behavior: everything inline in the current process, no lock, no subprocess — verified to produce bit-identical CSVs to the isolated path. `--lock_dir` defaults to the Docker path `/app/lock_dir`; pass a writable directory to run locally. `eval_timevae.run_phase_body` is the single phase-dispatch point — the subprocess wrapper calls it too, so inline and isolated cannot drift.

**Defaults are the A6 arm** (`--hidden_dims 50 100 200 --batch_size 16 --channel_wt "consumption=20,generation=5" --gen_channels <the 6 endogenous> --cond_dim 32`), the best configuration measured on the contiguous held-out block. `--channel_wt none` / `--gen_channels none` turn those off (`_disabled()` treats `""` and `"none"` as absent). Because of this, **the arm definitions in `generate_timevae_experiments.py` state all four axes explicitly** (`UNCONDITIONAL`, `NO_CHANNEL_WT`, `BIG_ARCH`, `REF_ARCH`) instead of inheriting defaults — otherwise changing a default would silently redefine what a past arm meant. Verified: A0–A6 resolve to the same values as before the defaults changed, and the cached checkpoints still hit (the hash covers resolved values, not defaults). HPO: `--optimize --n_trials --hpo_epochs --hpo_r_epochs --hpo_n_synthetic` (HPO overrides the generator hparams and uses the `-opt` generator for eval; its space now includes the two mean terms separately plus `cons_wt`). Examples: `python eval_timevae.py --loc austin --filter_bs 661 --model_name lstm --mode both --timevae_epochs 200 --r_epochs 20`; with channel weights: `... --channel_wt "consumption=20,generation=5"`; with seasonality: `... --trend_poly 3 --custom_seas "24x4" --align_to_day`. `--mode baseline` (train real → test real) is a useful reference: its error should be ≤ TSTR/TRTS. **Export `CUBLAS_WORKSPACE_CONFIG=:4096:8`** or the deterministic-algorithms setting crashes the backward pass on GPU.

#### Running the generative matrix for a whole location

`generate_timevae_experiments.py` builds the compose for the matrix **clients × arms × seeds**, mirroring `generate_simulation.py`:

```bash
python generate_timevae_experiments.py --loc austin --model_name lstm --arms A6 --seeds 0 --gpu_slots 3
docker compose -f docker-compose.timevae.austin.lstm.yml up --build
```

Omitting `--filter_bs` covers **every** client of the location (`get_available_clients_location`); pass a list (`--filter_bs 661 8156`) to restrict it. Every service mounts the same host `./lock_dir` so the `flock` is genuinely cross-container, and gets `--gpu_slots` — so `--gpu_slots 3` means at most 3 of the 25 containers train at once. The runs of a **single client** are chained with `service_completed_successfully`; different clients start in parallel. So the container count is the number of clients, not the number of runs — without that, a full 7 arms × 3 seeds × 25 clients matrix would try to start 525 containers just to leave them all blocked on the lock. Output file is `docker-compose.timevae.<loc>.<model>.yml`.

`run_all_timevae.sh` is the batch driver, in the mold of `run_all_models.sh` (same arg parsing, `.env` sourcing, `--notify`, `trap cleanup EXIT`): it sweeps models, generating and running one compose each.

```bash
bash run_all_timevae.sh -loc austin -gpu_slots 3 [-models rnn lstm gru] [-arms A6] [-seeds 0] \
  [-timevae_epochs 200] [-r_epochs 20] [-gen_holdout_days 0] [-no_plots] [-notify]
```

Unlike the FL script it **cannot** use `--exit-code-from` / `--abort-on-container-exit`: there is no coordinator service like `fl-server`, just N sibling clients, and either flag would tear down the other 24 as soon as the first finished. `docker compose up` therefore exits 0 even when a client failed, so the script collects the per-container exit codes afterwards (`compose ps -aq | xargs docker inspect -f '{{.Name}} {{.State.ExitCode}}'`) and reports `FAILED (n)` with the offending container names.

### Aggregation (`src/base/aggregation_strategy.py`, `src/utils/aggregation_functions.py`)

`Aggregator.aggregate(weights_list, current_model)` dispatches on `self.alg`. Weighted/simple/median variants are stateless; `fedadagrad`, `fedyogi`, `fedadam`, `fedavgm` are **stateful** — they persist momentum/variance vectors (`m_t`, `v_t`, `momentum_vector`) on the `Aggregator` instance across rounds. `fedprox` uses plain FedAvg aggregation on the server; its proximal `mu` term is applied **client-side** in `ClientLearning.train`.

### Forecasting Models (`src/models/`)

`RNN`, `LSTM`, `GRU`, `CNN` all take `(batch, lags, input_dim)` and output `output_dim` steps; built via `get_model()` in `src/utils/functions.py` (hidden size 128, 1 layer by default, `matrix_rep=True`). `input_dim`/`output_dim` come from each client's metadata, computed at preprocessing time.

**TimeVAE** (`src/models/timeVAE/`) is a variational autoencoder for time series — it is **never** the forecasting model. It serves two distinct roles: (1) **client selection**, producing latent signatures for clustering (`seq_len=num_lags`, feat on `X` only), cached at `etc/TimeVAE/<loc>/ckpt/{cid}-latent_dim_{d}.pth`; and (2) **generative evaluation** (TSTR/TRTS via `eval_timevae.py`), modelling the full multivariate window incl. the target (`seq_len=num_lags+output_dim`), cached at `etc/TimeVAE/<loc>/ckpt/{cid}-gen-{split}-latent_dim_{d}.pth`. The two checkpoints are not interchangeable.

### Dataset Pipeline (`src/dataset/`)

`participant_preprocessing.py` (`ParticipantData`) turns raw Pecan Street CSVs into feature frames (grid→`consumption`, solar→`generation`, temporal fields, weather join via `dataset/pecanstreet/weather_data/open-meteo-<loc>.csv`). `processing.py` (`Processing`, subclass of `Data`) handles NaN imputation, train/val split, scaling (default `minmax` to `[-1, 1]`), time-lag generation (default `num_lags=96` = 24h at 15-min sampling), and reshaping to `(samples, lags, features)`. The single default target is `consumption`. `LocalFileDataset` (`src/data.py`) just loads the resulting `.npy` tensors.

### Output Structure (all under `etc/`, gitignored)
- `etc/fl/server/ckpt/<Model>/` — best global checkpoints (`.pth`, keyed by loss/round)
- `etc/fl/local/ckpt/<Model>/<cid>/` — per-client local checkpoints
- `etc/fl/logs/<Model>/` — `history_simulation.pkl`, per-round local loss `.npy`, `blockchain_ledger.jsonl`
- `etc/fl/results/<Model>/` — final `global_model_cids_tests.pkl`
- `etc/TimeVAE/<loc>/` — cached TimeVAE checkpoints + logs
- `optuna_db/` — per-model SQLite Optuna studies

### Hyperparameter Optimization

With `--optimize_clients`, the server keeps a **per-client Optuna study** (`study_<cid>` in `optuna_db/fl_simulation_<Model>.db`). Before each training assignment it `ask()`s for `lr`, `batch_size`, `optimizer` (+ `fedprox_mu` when aggregation is `fedprox`), ships them to the client, and `tell()`s the study the returned **validation loss** after the update arrives.

## Non-obvious behaviors & gotchas

These are things that will bite during new implementations — verify against the code before relying on them:

- **`fednova` is broken via CLI.** `Aggregator.aggregate()` matches the key `"fednova_aggregate"`, but `get_params()`/`__repr__()` use `"fednova"`. Passing `--aggregation fednova` silently returns empty weights; passing `fednova_aggregate` raises `NotImplementedError` in `__repr__`. Fix the key mismatch before using FedNova.
- **Aggregator hyperparameters are initialized as a side effect of `__repr__()`.** `self.mu/rho/beta_1/eta/tau/...` are only set inside `Aggregator.__repr__`. It works today only because `FLServerState.__init__` logs `repr(self.aggr_strategy)` at startup. Do **not** remove that log call, or move the assignments into `__init__`, or stateful aggregators will `AttributeError` in `aggregate()`.
- **`weekly-representativeness` is now wired end-to-end** (branch `feature/timevae-selection`). The server sends `latent_mode="weekly"` at `INITIAL_EVAL`, the client's `get_latent_space(..., mode="weekly", samples_per_week=...)` emits a per-week dict `{"week_N": vector}`, and `TimeVAEWeeklyRepresentativeSelection.select()` indexes it (clamping `week_idx` to the max available week). See "TimeVAE selection — end-to-end flow" above. A client whose data yields fewer weeks than a given round's window is skipped for that window; if too few remain it still falls back to random.
- **`--optimize` across parallel clients relies on `optuna_db/` NOT being mounted.** `optimize_generative_timevae` uses one SQLite file per location (`optuna_db/timevae_generation_<loc>.db`), separated only logically by `study_name` (`gen_study_<cid>_<split>`). The TimeVAE compose deliberately mounts only `./etc`, `./lock_dir` and `./dataset`, so each container gets a private, ephemeral DB and there is no contention. Mounting `optuna_db/` to persist the studies would put N concurrent writers on one SQLite file (`database is locked`) — switch to a per-client DB file first.
- **`RoundRobinSelection` exists but is unreachable.** It's defined in `selection_strategy.py` but not wired into `get_select_strategy()` in `app-server.py`; only `random`/`fixed-representativeness`/`weekly-representativeness` are selectable.
- **Determinism requires a CUBLAS env var on GPU.** `seed_all()` calls `torch.use_deterministic_algorithms(True)`. On CUDA this needs `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set only in the *client* Docker env in `generate_simulation.py`). Local GPU runs may crash without exporting it.
- **WandB logging is entirely commented out.** Every `wandb.init/log` block in `fl_manager.py` and `client_learning.py` is disabled, even though `WANDB_*` args/env are still wired. Re-enabling means uncommenting those blocks, not just setting env vars.
- **The evaluate path no longer recomputes the latent space** (branch `feature/timevae-selection`). `app-client.py` now uses the `latent_space` returned by `ProcessExecutor.run_evaluate` (computed once in the isolated subprocess under `GPULock`); the old main-process recompute was removed. The subprocess wrappers in `process_executor.py` also call `seed_all(args.seed)` so TimeVAE first-time training / model init is reproducible across the `spawn` boundary.
- **`.env` is committed to git** (despite `.gitignore` listing `*.env`, because it was already tracked). It previously held a live `WANDB_API_KEY`, which has since been removed from the file — but the key is still recoverable from git history and must be treated as compromised (revoke/rotate it on wandb.ai; deleting it from the current file does not undo the leak). Do not add new secrets to a tracked file without accepting the same risk.

## Environment Variables

`.env` / Docker environment:
- `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_GROUP` — WandB integration (wired but currently disabled in code); currently unset
- `NOTIFY_WEBHOOK_URL` — Slack- or Discord-compatible incoming webhook URL used by `src/utils/notifier.py` to post a message when a simulation finishes/is interrupted/crashes. **Opt-in only**: nothing is sent unless `--enable_notifications` (`app-server.py`, `generate_simulation.py`) or `--notify` (`run_all_models.sh`, `run_all_timevae.sh`) is passed, even if this var is set. Both scripts source `.env` automatically; running `docker compose up` directly (outside that script) requires exporting it into the shell first, since compose's `${...}` substitution reads from the invoking shell's environment, not the container's.
- `PYTORCH_ALLOC_CONF=expandable_segments:True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS`, `MALLOC_ARENA_MAX` — GPU stability/determinism knobs set in the generated Docker Compose (`CUDA_LAUNCH_BLOCKING=1` is intentionally not set by default — it forces synchronous kernel launches and was killing training throughput; set it manually only when debugging an async CUDA error)
