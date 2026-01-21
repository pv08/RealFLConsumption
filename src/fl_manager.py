import numpy as np
from logging import INFO, WARNING

from torch.xpu import memory_stats_as_nested_dict

from src.models.rnn import RNN
from src.utils.logger import log

class FLServerState:
    def __init__(self, strategy, required_clients=5, clients_per_round=2, max_rounds=10):
        self.global_model = RNN(device='cpu', input_dim=1)
        self.global_weights = [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]
        self.phase = "WAITING_CLIENTS" # WAITING_CLIENTS, INITIAL_EVAL, TRAINING, GLOBAL_EVAL
        self.strategy = strategy
        self.required_clients = required_clients
        self.clients_per_round = clients_per_round
        self.max_rounds = max_rounds
        self.simulation_over = False
        # Estado do Sistema
        self.current_round = 0
        self.registered_clients = set()
        self.selected_clients = set()
        self.round_in_progress = False
        self.pending_messages = []
        self.updates_received = {}  # {client_id: weights}
        self.evaluations_received = {}
        # Modelo Global (Simulado - inicialize com a arquitetura real)
        # Exemplo simples de pesos para teste


    def register_client(self, client_id):
        if client_id not in self.registered_clients:
            log(INFO, f"New client assigned: {client_id}")
            self.registered_clients.add(client_id)

        if self.phase == "WAITING_CLIENTS" and len(self.registered_clients) >= self.required_clients:
            self.phase = "INITIAL_EVAL"
            self._notify_pending_clients()

    def _add_to_waitlist(self, message_obj):
        """Guarda a conexão para usar depois"""
        if message_obj not in self.pending_messages:
            log(INFO, f"Waiting more clients to start training (Long Polling)...")
            if message_obj is None:
                print()
            self.pending_messages.append(message_obj)

    def check_task(self, client_id, message_obj=None):
        """
        Agora aceita o objeto da mensagem para guardá-lo se necessário.
        """
        if message_obj is None:
            print()
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

    def receive_metrics(self, client_id, metrics):
        if client_id not in self.evaluations_received:
            self.evaluations_received[client_id] = metrics
            log(INFO, f"Metrics: {metrics} received from client {client_id} on fase {self.phase}")

            # Se todos avaliaram
        if len(self.evaluations_received) >= self.required_clients:
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
        self.phase = "TRAINING"

        log(INFO, f"Starting round {self.current_round}...")
        log(INFO, f"Clients selected: {self.selected_clients}")
        # Acorda todo mundo. Quem for selecionado treina, quem não for, volta a esperar.
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