import random
import numpy as np
from abc import ABC, abstractmethod
from src.utils.logger import log
from logging import INFO, WARNING
from src.base.clustering import (
    BaseClustering,
    euclidean_distance,
    squared_euclidean_distance,
    manhattan,
    cosine_similarity,
    hassanat_distance,
    minkowski_distance,
    chebyshev_distance,
    canberra_distance
)


class BaseSelectionStrategy(ABC):
    @abstractmethod
    def select(self, registered_clients, num_required, client_metadata=None):
        pass


class TimeVAE(BaseSelectionStrategy):
    def __init__(self, min_cluster_size=1, n_bootstrap=50):
        self.min_cluster_size = min_cluster_size
        self.n_bootstrap = n_bootstrap
        self.cached_representatives = None

        self.available_distances = {
            'euclidean': euclidean_distance,
            'euclidean-squared': squared_euclidean_distance,
            'manhattan': manhattan,
            'cosine-similarity': cosine_similarity,
            'hassanat': hassanat_distance,
            'minkowski': minkowski_distance,
            'chebyshev': chebyshev_distance,
            'canberra': canberra_distance
        }

    def select(self, registered_clients, num_required, client_metadata=None):
        available_cids = list(registered_clients.keys())
        if len(available_cids) <= num_required:
            return available_cids

        # Retorna o cache se já tivermos calculado os representantes
        if self.cached_representatives is not None:
            log(INFO, "Usando comitê de representantes em cache (Fixo).")
            return self.cached_representatives

        log(INFO, "Primeira execução: Calculando clientes representativos baseados no espaço latente global...")

        latents_list = []
        valid_cids = []

        for cid in available_cids:
            client_data = registered_clients[cid]
            if 'latent_space' in client_data:
                latents_list.append(client_data['latent_space'])
                valid_cids.append(cid)
            else:
                log(WARNING, f"Cliente {cid} ignorado: 'latent_space' ausente.")

        if len(valid_cids) <= num_required:
            return valid_cids

        X_latents = np.array(latents_list)

        # Avaliação de métricas
        best_metric_name = None
        best_mean_stability = -np.inf

        for metric_name, dist_fn in self.available_distances.items():
            try:
                clustering_model = BaseClustering(
                    n_clusters=num_required, distance_fn=dist_fn, min_cluster_size=self.min_cluster_size
                )
                results = clustering_model.centroid_stability(X_latents, n_bootstrap=self.n_bootstrap)

                if results["mean_stability"] > best_mean_stability:
                    best_mean_stability = results["mean_stability"]
                    best_metric_name = metric_name
            except Exception:
                pass

        if best_metric_name is None:
            log(WARNING, "Falha na clusterização. Retornando randômico.")
            return np.random.choice(valid_cids, num_required, replace=False).tolist()

        log(INFO, f"Métrica ótima selecionada: {best_metric_name} (Estabilidade: {best_mean_stability:.4f})")

        # Clusterização final
        best_dist_fn = self.available_distances[best_metric_name]
        final_clustering = BaseClustering(
            n_clusters=num_required, distance_fn=best_dist_fn, min_cluster_size=self.min_cluster_size
        )
        final_clustering.fit_predict(X_latents)
        representative_idxs = final_clustering.get_representatives()

        selected_clients = [valid_cids[idx] for idx in representative_idxs]

        # Salva em cache para sempre
        self.cached_representatives = selected_clients
        return selected_clients

    def __repr__(self):
        return "TimeVAE Fixed Representative Selection"


class TimeVAEWeeklyRepresentativeSelection(BaseSelectionStrategy):
    def __init__(self, rounds_per_week=10, min_cluster_size=1, n_bootstrap=50):
        self.rounds_per_week = rounds_per_week
        self.min_cluster_size = min_cluster_size
        self.n_bootstrap = n_bootstrap

        self.round_counter = 0
        self.cached_representatives = None
        self.cached_week_key = None

        self.available_distances = {
            'euclidean': euclidean_distance,
            'euclidean-squared': squared_euclidean_distance,
            'manhattan': manhattan,
            'cosine-similarity': cosine_similarity,
            'hassanat': hassanat_distance,
            'minkowski': minkowski_distance,
            'chebyshev': chebyshev_distance,
            'canberra': canberra_distance
        }

    def select(self, registered_clients, num_required, client_metadata=None):
        available_cids = list(registered_clients.keys())
        if len(available_cids) <= num_required:
            return available_cids

        self.round_counter += 1

        # Descobre qual é a semana atual baseada na rodada
        week_idx = ((self.round_counter - 1) // self.rounds_per_week) + 1
        current_week_key = f"week_{week_idx}"

        # Verifica se podemos usar o cache
        if self.cached_representatives is not None and self.cached_week_key == current_week_key:
            log(INFO, f"Usando representantes em cache da {current_week_key}. (Rodada {self.round_counter})")
            return self.cached_representatives

        log(INFO,
            f"Nova janela detectada ({current_week_key}). Calculando clusters para a rodada {self.round_counter}...")

        latents_list = []
        valid_cids = []

        for cid in available_cids:
            client_data = registered_clients[cid]
            if 'latent_space' in client_data and current_week_key in client_data['latent_space']:
                latents_list.append(client_data['latent_space'][current_week_key])
                valid_cids.append(cid)
            else:
                log(WARNING, f"Cliente {cid} não possui dados latentes para {current_week_key} ou formato incorreto.")

        if len(valid_cids) <= num_required:
            return valid_cids

        X_latents = np.array(latents_list)

        # Avaliação de métricas
        best_metric_name = None
        best_mean_stability = -np.inf

        for metric_name, dist_fn in self.available_distances.items():
            try:
                clustering_model = BaseClustering(
                    n_clusters=num_required, distance_fn=dist_fn, min_cluster_size=self.min_cluster_size
                )
                results = clustering_model.centroid_stability(X_latents, n_bootstrap=self.n_bootstrap)

                if results["mean_stability"] > best_mean_stability:
                    best_mean_stability = results["mean_stability"]
                    best_metric_name = metric_name
            except Exception:
                pass

        if best_metric_name is None:
            log(WARNING, "Falha na clusterização semanal. Retornando randômico.")
            return np.random.choice(valid_cids, num_required, replace=False).tolist()

        log(INFO, f"Métrica ótima selecionada: {best_metric_name} (Estabilidade: {best_mean_stability:.4f})")

        # Clusterização final
        best_dist_fn = self.available_distances[best_metric_name]
        final_clustering = BaseClustering(
            n_clusters=num_required, distance_fn=best_dist_fn, min_cluster_size=self.min_cluster_size
        )
        final_clustering.fit_predict(X_latents)
        representative_idxs = final_clustering.get_representatives()

        selected_clients = [valid_cids[idx] for idx in representative_idxs]

        # Atualiza o cache para a nova semana
        self.cached_representatives = selected_clients
        self.cached_week_key = current_week_key
        return selected_clients

    def __repr__(self):
        return f"TimeVAE Weekly Representative Selection (Rounds/Week: {self.rounds_per_week})"

class RandomSelection(BaseSelectionStrategy):
    def select(self, registered_clients, num_required, client_metadata=None):
        # Transforma em lista para poder usar sample
        available = list(registered_clients)
        if len(available) < num_required:
            return available
        return random.sample(available, num_required)

    def __repr__(self):
        return "Random Selection"


class RoundRobinSelection(BaseSelectionStrategy):
    def __init__(self):
        self.last_index = 0

    def select(self, registered_clients, num_required, client_metadata=None):
        available = sorted(list(registered_clients))
        total = len(available)
        if total == 0: return []

        selected = []
        for _ in range(num_required):
            client = available[self.last_index % total]
            selected.append(client)
            self.last_index += 1

        return list(set(selected))  # Remove duplicatas se num_required > total

    def __repr__(self):
        return "Robin Selection"
