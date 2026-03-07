import copy
import os
import numpy as np
import torch as T
import pickle
import optuna
import wandb
from collections import defaultdict
from logging import INFO, WARNING
from typing import List, Dict, Tuple
from collections import OrderedDict
from src.utils.functions import mkdir_if_not_exists, get_model, seed_all
from src.structure.blockchain import Blockchain
from src.utils.logger import log

class FLServerState:
    def __init__(self, selection_strategy, aggr_strategy, required_clients=5, clients_per_round=2, max_rounds=10, optimize_clients: bool=True, wandb_config: dict=None, seed: int=0):
        self.global_model = None
        self.global_weights = None
        self.wandb_config = wandb_config or {}
        self.seed = seed
        seed_all(self.seed)
        # wandb.init(
        #     project=self.wandb_config.get('project', 'fl_simulation'),
        #     group=self.wandb_config.get('group', 'experiment_1'),
        #     name="server_orchestrator",
        #     job_type="server",
        #     config={
        #         "strategy": str(selection_strategy),
        #         "aggregation": str(aggr_strategy),
        #         "clients_per_round": clients_per_round,
        #         "required_clients": required_clients,
        #         "max_rounds": max_rounds
        #     },
        #     reinit=True
        # )
        self.phase = "WAITING_CLIENTS" # WAITING_CLIENTS, INITIAL_EVAL, TRAINING, GLOBAL_EVAL
        self.selection_strategy = selection_strategy
        self.aggr_strategy = aggr_strategy
        self.required_clients = required_clients
        self.clients_per_round = clients_per_round
        self.max_rounds = max_rounds
        self.simulation_over = False
        self.current_round = 0
        self.registered_clients = defaultdict()
        self.selected_clients = set()
        self.round_in_progress = False
        self.model_name = None
        self.pending_messages = []
        self.updates_received = {}
        self.evaluations_received = {}
        self.tests_received = {}

        self.optimize_clients = optimize_clients
        if self.optimize_clients:
            log(WARNING, f"[Optuna] Optimizing procedure activated: all clients available will receive hparams from Optuna")
        else:
            log(WARNING, f"No optimization procedure activated. Hparams will be fixed")
        self.client_studies = {}
        self.client_active_trials = {}

        self.history = defaultdict(list)
        self.best_loss, self.best_round = np.inf, -1
        log(INFO, f"Aggregation Algorithm: {repr(self.aggr_strategy)}")
        log(INFO, f"Client Selection Mechanism: {repr(self.selection_strategy)}")

    def _get_or_create_study(self, client_id):
        """Cria um estudo do Optuna para o cliente, se ainda não existir."""
        mkdir_if_not_exists("optuna_db/")

        storage_url = f"sqlite:///optuna_db/fl_simulation_{self.model_name}.db"

        study_name = f"study_{client_id}"

        if client_id not in self.client_studies:
            log(INFO, f"[Optuna] Creating new study for client {client_id}")
            self.client_studies[client_id] = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction="minimize",
                load_if_exists=True
            )
        return self.client_studies[client_id]

    def _get_hparams_from_optuna(self, client_id):
        """Pede ao Optuna uma nova configuração de hiperparâmetros."""
        study = self._get_or_create_study(client_id)

        # O Optuna gera um novo 'trial' (uma tentativa)
        trial = study.ask()

        # Definimos o espaço de busca
        hparams = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024]),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
        }

        # Otimiza o fedprox_mu apenas se a agregação for FedProx
        if self.aggr_strategy.alg == "fedprox":
            hparams["fedprox_mu"] = trial.suggest_float("fedprox_mu", 0.001, 0.1)

        # Guardamos o trial ativo para podermos reportar a 'loss' mais tarde
        log(INFO, f"[Optuna] Client {client_id} | Trial {trial.number} | New suggestion generated: {hparams}")
        self.client_active_trials[client_id] = trial

        return hparams


    @staticmethod
    def weighted_loss_avg(n_per_client: List[int], losses: List[float]) -> float:
        """Aggregates losses received from clients"""
        n = sum(n_per_client)
        weighted_losses = [n_k * loss for n_k, loss in zip(n_per_client, losses)]
        return sum(weighted_losses) / n

    @staticmethod
    def weighted_metrics_avg(n_per_client: List[int], metrics_per_client: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        n = sum(n_per_client)
        metrics = dict()
        for cid in metrics_per_client:
            for metric in metrics_per_client[cid]:
                if metric not in metrics:
                    metrics[metric] = []
                metrics[metric].append(metrics_per_client[cid][metric])
        weighted_metrics = dict()
        for metric in metrics:
            weighted_metric = [n_k * m for n_k, m in zip(n_per_client, metrics[metric])]
            weighted_metrics[metric] = sum(weighted_metric) / n
        return weighted_metrics


    def _get_parameters(self, model):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def _define_global_model_architecture(self):
        assert all(set(v['model_name'] for v in self.registered_clients.values())), f"Make sure all the clients have the same architecture"
        _tmp_obj = next(iter(self.registered_clients.values()))
        self.global_model = get_model(device=_tmp_obj["device"], model=_tmp_obj["model_name"], input_dim=_tmp_obj["input_dim"],
                  out_dim=_tmp_obj["output_dim"],
                  lags=_tmp_obj["lags"])
        self.global_weights = [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]
        self.model_name = type(self.global_model).__name__
        self.ledger = Blockchain(log_dir=f'etc/fl/logs/{self.model_name}/')
        log(INFO, f"Global model architecture defined as {type(self.global_model).__name__}")

    def register_client(self, client_id, message_obj):
        if client_id not in self.registered_clients:
            log(INFO, f"Client {client_id} assigned using {message_obj['model_name'].upper()} architecture -> {list(self.registered_clients.keys())}")
            self.registered_clients[client_id]= message_obj

        if self.phase == "WAITING_CLIENTS" and len(self.registered_clients) >= self.required_clients:
            self._define_global_model_architecture()
            self.phase = "INITIAL_EVAL"
            self._notify_pending_clients()

    def _add_to_waitlist(self, message_obj):
        """Guarda a conexão para usar depois"""
        if message_obj not in self.pending_messages:
            log(INFO, f"Waiting more clients to start training (Long Polling)...")
            self.pending_messages.append(message_obj)

    def check_task(self, client_id, message_obj=None):
        """
        Agora aceita o objeto da mensagem para guardá-lo se necessário.
        """
        if self.simulation_over:
            return "stop", {"architecture": self.global_model, "weights": self.global_weights}

        # Espera até que a quantidade de clientes seja o suficiente
        if self.phase == "WAITING_CLIENTS":
            self._add_to_waitlist(message_obj)
            return "defer", None

        if self.phase in ["INITIAL_EVAL", "GLOBAL_EVAL"]:
            if client_id not in self.evaluations_received:
                req_latent_space = (self.phase == "INITIAL_EVAL" and "TimeVAE" in type(self.selection_strategy).__name__)

                return "evaluate", {
                    "phase": self.phase,
                    "architecture": self.global_model,
                    "weights": self.global_weights,
                    "req_latent_space": req_latent_space
                }
            else:
                self._add_to_waitlist(message_obj)
                return "defer", None

        if self.phase == "TRAINING":
            if client_id in self.selected_clients:
                if client_id not in self.updates_received:
                    hparams = {}
                    if self.optimize_clients:
                        hparams = self._get_hparams_from_optuna(client_id)
                    data = {
                        "current_round": self.current_round,
                        "weights": self.global_weights,
                        "hparams": hparams
                    }
                    return "train", data
                else:
                    self._add_to_waitlist(message_obj)
                    return "defer", None
            else:
                # NÃO SELECIONADO: Fica em Long Polling até a fase mudar para GLOBAL_EVAL
                self._add_to_waitlist(message_obj)
                return "defer", None
        return "defer", None

    def _log_and_save_evaluation_procedure(self):
        _losses = [v["loss"] for k, v in self.evaluations_received.items()]
        _instances = [v["instances"] for k, v in self.evaluations_received.items()]
        _metrics = {k: v["metrics"] for k, v in self.evaluations_received.items()}
        _weighted_losses = self.weighted_loss_avg(_instances, _losses)
        _weighted_metrics = self.weighted_metrics_avg(_instances, _metrics)
        self.history["evaluation"].append({"round": self.current_round, "round_eval": self.evaluations_received, "weighted_loss": _weighted_losses,
                                            "weighted_metrics": _weighted_metrics})
        _round_losses = [r['weighted_loss'] for r in self.history["evaluation"]]
        if _round_losses[-1] <= self.best_loss:
            self.best_loss = _round_losses[-1]
            self.best_round = self.current_round
            mkdir_if_not_exists(f'etc/fl/server/ckpt/{self.model_name}')
            T.save(self.global_model.state_dict(), f"etc/fl/server/ckpt/{self.model_name}/global_model_loss-{self.best_loss}_round-{self.best_round}.pth")

        log_dict = {
            "server/global_loss": _weighted_losses,
            "server/round": self.current_round,
            "server/best_loss": self.best_loss
        }
        for metric_name, metric_val in _weighted_metrics.items():
            log_dict[f"server/global_{metric_name}"] = metric_val

        # wandb.log(log_dict)

    def receive_test(self, client_id, results):
        if client_id not in self.tests_received:
            self.tests_received[client_id] = results
            log(INFO, f"Results from client {client_id} received ")

            # Se todos avaliaram
        if len(self.tests_received) >= self.required_clients:
            mkdir_if_not_exists(f'etc/fl/results/{self.model_name}/')
            with open(f'etc/fl/logs/{self.model_name}/history_simulation.pkl', "wb") as f:
                pickle.dump(self.history, f)
            log(INFO, f"Simulation history saved on etc/fl/logs/{self.model_name}/history_simulation.pkl")
            with open(f'etc/fl/results/{self.model_name}/global_model_cids_tests.pkl', "wb") as f:
                pickle.dump(self.tests_received, f)
                log(INFO, f"Simulation testing saved on etc/fl/results/{self.model_name}/global_model_cids_tests.pkl")

    def receive_metrics(self, client_id, metrics):
        if metrics["latent_space"] is not None:
            latent_space = metrics.pop("latent_space")
            if client_id in self.registered_clients:
                self.registered_clients[client_id]["latent_space"] = latent_space

        if client_id not in self.evaluations_received:
            self.evaluations_received[client_id] = metrics
            log(INFO, f"Metrics from client {client_id} on fase {self.phase}: {metrics} ")

            # Se todos avaliaram
        if len(self.evaluations_received) >= self.required_clients:
            self._log_and_save_evaluation_procedure()
            if self.phase == "INITIAL_EVAL":
                log(INFO, "Initial evaluation completed. Turning to TRAIN")
                self._start_training_phase()
                self.evaluations_received = {}
            elif self.phase == "GLOBAL_EVAL":
                self.current_round += 1
                if self.current_round >= self.max_rounds:
                    self.simulation_over = True
                    self._notify_all_stop()
                else:
                    self._start_training_phase()  # Próxima rodada, novo sorteio
                    self.evaluations_received = {}

    def _log_and_save_update_procedure(self):
        _train_losses = [v["train_loss"] for k, v in self.updates_received.items()]
        _train_instances = [v["train_instances"] for k, v in self.updates_received.items()]
        _train_metrics = {k: v["train_metrics"] for k, v in self.updates_received.items()}
        _train_weighted_losses = self.weighted_loss_avg(_train_instances, _train_losses)
        _train_weighted_metrics = self.weighted_metrics_avg(_train_instances, _train_metrics)


        _val_losses = [v["val_loss"] for k, v in self.updates_received.items()]
        _val_instances = [v["val_instances"] for k, v in self.updates_received.items()]
        _val_metrics = {k: v["val_metrics"] for k, v in self.updates_received.items()}

        _val_weighted_losses = self.weighted_loss_avg(_val_instances, _val_losses)
        _val_weighted_metrics = self.weighted_metrics_avg(_val_instances, _val_metrics)

        self.history["update"].append({"round": self.current_round, "round_update": self.updates_received, "train_weighted_loss": _train_weighted_losses,
                                            "train_weighted_metrics": _train_weighted_metrics, "val_weighted_loss": _val_weighted_losses,
                                            "val_weighted_metrics": _val_weighted_metrics})

        mkdir_if_not_exists(f'etc/fl/local/ckpt/{self.model_name}/')
        mkdir_if_not_exists(f'etc/fl/logs/{self.model_name}/local/')

        for k, v in self.updates_received.items():
            mkdir_if_not_exists(f'etc/fl/logs/{self.model_name}/local/{k}')
            with open(f'etc/fl/logs/{self.model_name}/local/{k}/fl_round_{self.current_round}_local_train_loss.npy', 'wb') as f:
                np.save(f, np.array(v["train_history"]))
            with open(f'etc/fl/logs/{self.model_name}/local/{k}/fl_round_{self.current_round}_local_val_loss.npy', 'wb') as f:
                np.save(f, np.array(v["val_history"]))

            _tmp_model = copy.deepcopy(self.global_model)
            params_dict = zip(_tmp_model.state_dict().keys(), v["params"])
            state_dict = OrderedDict({k: T.Tensor(v) for k, v in params_dict})
            _tmp_model.load_state_dict(state_dict, strict=True)
            mkdir_if_not_exists(f'etc/fl/local/ckpt/{self.model_name}/{k}')
            T.save(_tmp_model.state_dict(),
                   f"etc/fl/local/ckpt/{self.model_name}/{k}/local_model_loss-{v['val_loss']}_round-{self.current_round}.pth")

            log(INFO, f"etc/fl/local/ckpt/{self.model_name}/{k}/local_model_loss-{v['val_loss']}_round-{self.current_round}.pth")

    def receive_update(self, client_id, client_res):
        """Recebe pesos treinados e verifica agregação."""
        model_params, train_history, num_train, train_loss, train_metrics, val_history, num_val, val_loss, val_metrics, time_spent = client_res

        if not self.ledger.add_block(client_id=client_id, model_weights=model_params):
            log(WARNING, f"Update from {client_id} rejected by Blockchain Protocol.")
            return False

        if self.optimize_clients and client_id in self.client_active_trials:
            trial = self.client_active_trials[client_id]
            study = self.client_studies[client_id]
            study.tell(trial, val_loss)

            log(INFO, f"[Optuna] Client {client_id} | Trial {trial.number} finished with val_Loss: {val_loss:.6f}")
            log(INFO, f"[Optuna] Client {client_id} | Best val_loss so far: {study.best_value:.6f}")
            log(INFO, f"[Optuna] Client {client_id} | Best hparams so far: {study.best_params}")

            del self.client_active_trials[client_id]

        if client_id not in self.updates_received:
            self.updates_received[client_id] = {"params": model_params, "train_history": train_history, "train_instances": num_train, "train_loss": train_loss,
                                                "train_metrics": train_metrics, "val_history":val_history, "val_instances": num_val,
                                                "val_loss": val_loss, "val_metrics": val_metrics, "time_spent": time_spent}
            log(INFO, f"Update received from client {client_id}")

        # Se todos treinaram, agrega e vai para avaliação global
        if len(self.updates_received) >= len(self.selected_clients):
            self._log_and_save_update_procedure()
            log(INFO, "All selected clients trained.")
            weight_list = [(v["params"], v["train_instances"]) for k, v in self.updates_received.items()]
            self.global_weights = self._aggregate_models(weight_list)
            self.updates_received = {}
            self.phase = "GLOBAL_EVAL"
            self._notify_pending_clients()
        return True


    def _notify_all_stop(self):
        """Acorda TODOS os clientes na fila de espera e manda parar"""
        log(WARNING, f"Sending stop signal to {len(self.pending_messages)} waiting.")
        for msg in self.pending_messages:
            # Envia a ordem de parada
            content = {"action": "stop", "data": {"architecture": self.global_model, "weights": self.global_weights}}
            msg.trigger_delayed_response(content)

        # Limpa a lista
        self.pending_messages = []

    def _start_training_phase(self):
        """Sorteia os clientes que vão participar desta rodada de treino."""
        self.selected_clients = set(
            self.selection_strategy.select(self.registered_clients, self.clients_per_round)
        )

        selection_log = {"server/round": self.current_round}

        self.history["client_selection"].append({"round": self.current_round, "clients": list(self.selected_clients)})
        for cid in self.registered_clients.keys():
            selection_log[f"selection/client_{cid}"] = 1 if cid in selection_log else 0

        self.phase = "TRAINING"

        # wandb.log(selection_log)

        log(INFO, f"Starting round {self.current_round}...")
        log(INFO, f"Clients selected: {self.selected_clients}")
        self._notify_pending_clients()


    def _notify_pending_clients(self):
        """Acorda os clientes em Long Polling para a próxima tarefa."""
        log(INFO, f"Waking up {len(self.pending_messages)} clients...")
        temp_list = self.pending_messages[:]
        self.pending_messages = []
        for msg in temp_list:
            # Pegamos o ID do cliente que está associado a esta conexão
            # (Assumindo que você passou o client_id na request de check_in)
            client_id = msg.request['content'].get("client_id")

            # Reavaliamos a tarefa para este cliente específico
            task, data = self.check_task(client_id, msg)
            log(INFO, f"Telling client {client_id} to {task}")
            if task != "defer":
                # CHAMADA CRUCIAL: Envia os dados para o socket que estava esperando
                msg.trigger_delayed_response({"action": task, "data": data})

    def _aggregate_models(self, weights_list: List[Tuple[List[np.ndarray], int]]):
        new_weights = self.aggr_strategy.aggregate(weights_list, self.global_model)
        return new_weights
