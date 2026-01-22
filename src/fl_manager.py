import copy
import numpy as np
import torch as T

from collections import defaultdict
from logging import INFO, WARNING
from typing import List, Dict

from src.utils.functions import mkdir_if_not_exists
from src.utils.logger import log

class FLServerState:
    def __init__(self, strategy, required_clients=5, clients_per_round=2, max_rounds=10):
        self.global_model = None
        self.global_weights = None
        self.phase = "WAITING_CLIENTS" # WAITING_CLIENTS, INITIAL_EVAL, TRAINING, GLOBAL_EVAL
        self.strategy = strategy
        self.required_clients = required_clients
        self.clients_per_round = clients_per_round
        self.max_rounds = max_rounds
        self.simulation_over = False
        # Estado do Sistema
        self.current_round = 0
        self.registered_clients = defaultdict()
        self.selected_clients = set()
        self.round_in_progress = False
        self.model_name = None
        self.pending_messages = []
        self.updates_received = {}  # {client_id: weights}
        self.evaluations_received = {}
        # Modelo Global (Simulado - inicialize com a arquitetura real)
        self.history = defaultdict(list)
        self.best_loss, self.best_round = np.inf, -1

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
        assert all(set(v for v in self.registered_clients.values())), f"Make sure all the clients have the same architecture"
        self.global_model = copy.deepcopy(next(iter(self.registered_clients.values())))
        # Se todos os modelo são iguais, faça uma média
        weight_list = [self._get_parameters(client) for client in self.registered_clients.values()]
        self.global_weights = self._aggregate_models(weight_list)
        self.model_name = type(self.global_model).__name__
        log(INFO, f"Global model architecture defined as {type(self.global_model).__name__}")

    def register_client(self, client_id, architecture):
        if client_id not in self.registered_clients:
            log(INFO, f"Client {client_id} assigned using {type(architecture).__name__} architecture ")
            self.registered_clients[client_id]=architecture

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
            return "stop", None

        # Espera até que a quantidade de clientes seja o suficiente
        if self.phase == "WAITING_CLIENTS":
            self._add_to_waitlist(message_obj)
            return "defer", None

        if self.phase in ["INITIAL_EVAL", "GLOBAL_EVAL"]:
            if client_id not in self.evaluations_received:
                return "evaluate", {"architecture": self.global_model, "weights": self.global_weights}
            else:
                self._add_to_waitlist(message_obj)
                return "defer", None

        if self.phase == "TRAINING":
            if client_id in self.selected_clients:
                if client_id not in self.updates_received:
                    return "train", self.global_weights
                else:
                    # Já enviou o update, espera a próxima fase
                    self._add_to_waitlist(message_obj)
                    return "defer", None
            else:
                # NÃO SELECIONADO: Fica em Long Polling até a fase mudar para GLOBAL_EVAL
                self._add_to_waitlist(message_obj)
                return "defer", None

        return "defer", None

    def _log_and_save(self, metrics):
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
            T.save(self.global_model.state_dict(), f"etc/fl/server/ckpt/{self.model_name}/global_model_loss-{round(self.best_loss, 4)}_round-{self.best_round}.pth")

    def receive_metrics(self, client_id, metrics):
        if client_id not in self.evaluations_received:
            self.evaluations_received[client_id] = metrics
            log(INFO, f"Metrics from client {client_id} on fase {self.phase}: {metrics} ")

            # Se todos avaliaram
        if len(self.evaluations_received) >= self.required_clients:
            self._log_and_save(metrics)
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

    def receive_update(self, client_id, weights):
        """Recebe pesos treinados e verifica agregação."""
        if client_id not in self.updates_received:
            self.updates_received[client_id] = weights
            log(INFO, f"Update received from client {client_id}")

        # Se todos treinaram, agrega e vai para avaliação global
        if len(self.updates_received) >= len(self.selected_clients):
            log(INFO, "All selected clients trained.")
            self.global_weights = self._aggregate_models(list(self.updates_received.values()))
            self.updates_received = {}
            self.phase = "GLOBAL_EVAL"
            self._notify_pending_clients()


    def _notify_all_stop(self):
        """Acorda TODOS os clientes na fila de espera e manda parar"""
        log(WARNING, f"Sending stop signal to {len(self.pending_messages)} waiting.")

        for msg in self.pending_messages:
            # Envia a ordem de parada
            content = {"action": "stop"}
            msg.trigger_delayed_response(content)

        # Limpa a lista
        self.pending_messages = []

    def _start_training_phase(self):
        """Sorteia os clientes que vão participar desta rodada de treino."""
        self.selected_clients = set(
            self.strategy.select(self.registered_clients, self.clients_per_round)
        )
        self.history["client_selection"].append({"round": self.current_round, "clients": list(self.selected_clients)})
        self.phase = "TRAINING"

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

    def _aggregate_models(self, weights_list):
        """
        Média Simples (FedAvg). 
        Na prática, converta dicts para Tensors, faça a média e salve.
        """
        log(INFO, "Aggregating models using (FedAvg)...")
        """Média Simples (FedAvg) de listas de arrays NumPy."""
        new_weights = []
        for layer_idx in range(len(weights_list[0])):
            layer_avg = np.mean([w[layer_idx] for w in weights_list], axis=0)
            new_weights.append(layer_avg)
        return new_weights