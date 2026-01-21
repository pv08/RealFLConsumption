import random
from abc import ABC, abstractmethod


class BaseSelectionStrategy(ABC):
    @abstractmethod
    def select(self, registered_clients, num_required, client_metadata=None):
        pass


class RandomSelection(BaseSelectionStrategy):
    def select(self, registered_clients, num_required, client_metadata=None):
        # Transforma em lista para poder usar sample
        available = list(registered_clients)
        if len(available) < num_required:
            return available
        return random.sample(available, num_required)


class RoundRobinSelection(BaseSelectionStrategy):
    def __init__(self):
        self.last_index = 0

    def select(self, registered_clients, num_required, client_metadata=None):
        available = sorted(list(registered_clients))  # Ordenar é importante aqui
        total = len(available)
        if total == 0: return []

        selected = []
        for _ in range(num_required):
            client = available[self.last_index % total]
            selected.append(client)
            self.last_index += 1

        return list(set(selected))  # Remove duplicatas se num_required > total