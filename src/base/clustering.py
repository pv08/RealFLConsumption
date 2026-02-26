import numpy as np
import pandas as pd
import heapq
from logging import INFO
from collections import defaultdict
from src.utils.logger import log
from typing import Optional, Callable
from argparse import ArgumentParser
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cosine, canberra
from scipy.special import rel_entr
from sklearn.utils import resample
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment



def hassanat_distance(p1: np.ndarray, p2: np.ndarray):
    d =  (1 + np.minimum(p1, p2)) / (1 + np.maximum(p1, p2))
    return np.sum(1 - d)

def minkowski_distance(p1: np.ndarray, p2: np.ndarray, p: int = 3) -> float:
    r"""
    Calculate the euclidian distance.

    Follows the equation: $$d\left(x, y\right)=\left|\sum^{n}_{i\gets1}\left(x_i-y_i\right|^p\right)^{\frac{1}{p}}$$

    Args:
        p1 (float): Point 1.
        p2 (float): Point 2
        p (int): Sensitive point. Default = 3

    Returns:
        float: The distance
    """
    return np.power(np.sum(np.power(np.abs(p1 - p2), p)), 1 / p)



def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    r"""
    Calculate the euclidian distance.

    Follows the equation: $$d\left(x, y\right)=\sqrt{\sum^{n}_{i\gets1}\left(x_i-y_i\right)^2}$$

    Args:
        p1 (float): Point 1.
        p2 (float): Point 2

    Returns:
        float: The distance
    """
    return np.sqrt(np.sum((p1 - p2) ** 2))



def squared_euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    r"""
    Calculate the squared euclidian distance.

    Follows the equation: $$d\left(x, y\right)=\sum^{n}_{i\gets1}\left(x_i-y_i\right)^2$$

    Args:
        p1 (float): Point 1.
        p2 (float): Point 2

    Returns:
        float: The distance
    """
    return np.sqrt(np.sum((p1 - p2) ** 2)) ** 2



def manhattan(p1: np.ndarray, p2: np.ndarray) -> float:
    r"""
    Calculate the Manhattan distance

    Follows the equation: $$d\left(x, y\right)=\sum^{n}_{i\gets1}\left|x_i-y_i\right|$$

    Args:
        p1 (float): Point 1.
        p2 (float): Point 2

    Returns:
        float: The distance

    """
    return np.sum(np.abs(p1 - p2))



def cosine_similarity(p1: np.ndarray, p2: np.ndarray) -> float:
    r"""
    Calculate the Cosine distance

    Follows the equation: $$d\left(x, y\right)=1 - \frac{x\cdot y}{\left|x\right|\left|y\right|}$$

    Args:
        p1 (float): Point 1.
        p2 (float): Point 2

    Returns:
        float: The distance

    """
    return 1 - cosine(p1, p2)



def chebyshev_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    r"""
    Calculate the Cosine distance

    Follows the equation: $$d\left(x, y\right)=1 - \text{max}\left(\left|x-y\right|\right)$$

    Args:
        p1 (float): Point 1.
        p2 (float): Point 2

    Returns:
        float: The distance

    """

    return np.max(np.abs(p1 - p2))


def jensen_shannon_divergence(p1: np.ndarray, p2: np.ndarray) -> float:
    M = 0.5 * (p1 + p2)
    return np.sqrt(0.5 * (rel_entr(p1, M).sum() + rel_entr(p2, M).sum()))


def canberra_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return canberra(p1, p2)


class BaseClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters: int, distance_fn: Callable[[float, float], float], linkage: str='average', min_cluster_size: int=3):
        self.n_clusters = n_clusters
        self.distance_fn = distance_fn
        self.linkage = linkage
        self.min_cluster_size = min_cluster_size
        self.labels_ = None
        self.X_ = None

    def _compute_distance_matrix(self, X: np.ndarray):
        n_samples = X.shape[0]
        dist_matrix = np.zeros( (n_samples, n_samples) )
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = self.distance_fn(X[i], X[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        return dist_matrix


    def fit(self, X: np.ndarray, y: Optional[np.ndarray]=None):
        self.X_ = X
        dist_matrix =self._compute_distance_matrix(X)
        model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric='precomputed',
            linkage=self.linkage
        )

        labels = model.fit_predict(dist_matrix)
        self.labels_ = self._enforce_min_cluster_size(labels)
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray]=None, **kwargs):
        self.fit(X)
        return self.labels_

    def _enforce_min_cluster_size(self, labels):
        n_clients = len(labels)
        cluster_map = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_map[label].append(idx)

        reassigned = set()

        for label in sorted(cluster_map):
            indices = cluster_map[label]

            if len(indices) >= self.min_cluster_size:
                continue

            needed = self.min_cluster_size - len(indices)

            # Compute centroid of the current cluster (if not empty)
            if len(indices) == 0:
                print(f"[Warning] Cluster {label} is empty. Seeding from largest cluster.")
                cluster_centroid = None
            else:
                cluster_centroid = np.mean(self.X_[indices], axis=0)

            for _ in range(needed):
                # Find donor cluster with most clients (excluding self)
                donor_label = max(
                    (l for l in cluster_map if l != label and len(cluster_map[l]) > self.min_cluster_size),
                    key=lambda l: len(cluster_map[l]),
                    default=None
                )

                if donor_label is None:
                    raise ValueError("No valid donor cluster found with enough clients to steal from.")

                donor_indices = cluster_map[donor_label]

                # Select best candidate client from donor cluster
                if cluster_centroid is None:
                    donor_client = donor_indices[0]
                else:
                    donor_client = min(
                        donor_indices,
                        key=lambda idx: self.distance_fn(self.X_[idx], cluster_centroid)
                    )

                # Reassign client
                labels[donor_client] = label
                cluster_map[donor_label].remove(donor_client)
                cluster_map[label].append(donor_client)
                reassigned.add(donor_client)

                # Update centroid if needed
                if cluster_centroid is not None:
                    indices.append(donor_client)
                    cluster_centroid = np.mean(self.X_[indices], axis=0)

        return labels

    def get_representatives(self):
        if self.labels_ is None:
            raise RuntimeError(f"You need to fit first...")

        representatives = []
        for cluster_idx in range(self.n_clusters):
            indices = np.where(self.labels_ == cluster_idx)[0]
            cluster_points = self.X_[indices]

            centroid = np.mean(cluster_points, axis=0)

            closest_idx = min(indices, key=lambda idx: self.distance_fn(self.X_[idx], centroid))
            representatives.append(closest_idx)
        return np.array(representatives)

    def _compute_centroids(self, X, labels):
        """Helper to compute centroids of clusters."""
        centroids = []
        for cluster_idx in range(self.n_clusters):
            indices = np.where(labels == cluster_idx)[0]
            if len(indices) > 0:
                centroids.append(np.mean(X[indices], axis=0))
            else:
                centroids.append(None)
        return centroids

    def centroid_stability(self, X, n_bootstrap=50, random_state=0):
        """
        Evaluate centroid stability via bootstrapping.

        Returns dict with base centroids, bootstrap stability scores,
        mean stability, and std stability.
        """
        rng = np.random.RandomState(random_state)

        # Fit on original data
        base_labels = self.fit_predict(X)
        base_centroids = self._compute_centroids(X, base_labels)

        bootstrap_centroids = []

        for b in range(n_bootstrap):
            X_resampled = resample(X, replace=True, random_state=rng.randint(0, 1e6))
            labels = self.fit_predict(X_resampled)
            centroids = self._compute_centroids(X_resampled, labels)
            bootstrap_centroids.append(centroids)

        stability_scores = []
        valid_base_centroids = np.vstack([c for c in base_centroids if c is not None])

        for centroids in bootstrap_centroids:
            valid_centroids = np.vstack([c for c in centroids if c is not None])
            cost_matrix = cdist(valid_base_centroids, valid_centroids)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            stability_scores.append(cost_matrix[row_ind, col_ind].mean())

        return {
            "base_centroids": base_centroids,
            "stability_scores": stability_scores,
            "mean_stability": np.mean(stability_scores),
            "std_stability": np.std(stability_scores)
        }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--distance_fn', type=str, default='euclidian-squared', help="['hassanat', 'euclidean', 'manhattan', 'cosine-similarity', 'minkowski', 'chebyshev', 'euclidian-squared','canberra']")
    parser.add_argument('--location', type=str, default='austin',help="['austin', 'newyork', 'california', 'puertorico']")
    parser.add_argument('--latent_path', type=str, default='etc/TimeVAE/austin/results/clusters/client_latents_by_dim_austin.json')

    args =parser.parse_args()
    log(INFO, args)
    available_distance_fn = {'hassanat': hassanat_distance, 'euclidian': euclidean_distance,
                             'manhattan': manhattan,
                             'cosine-similarity': cosine_similarity,
                             'minkowski': minkowski_distance,
                             'chebyshev': chebyshev_distance,
                             'euclidian-squared': squared_euclidean_distance,
                             'canberra': canberra_distance}

    assert args.distance_fn in available_distance_fn, f"{args.distance_fn} distance function not available. Select one of the following: {available_distance_fn.keys()}."
    distance_fn = available_distance_fn[args.distance_fn]

    client_generators = np.load(args.latent_path, allow_pickle=True)[0]
    logs = []
    for k, v in client_generators.items():
        cids = np.array(list(v.keys()))
        latents = np.array([client['latents'] for client in v.values()])
        client_generators[k]['latents'] = latents

        custom_clustering = BaseClustering(n_clusters=5, distance_fn=distance_fn)
        results = custom_clustering.centroid_stability(latents, n_bootstrap=100)

        print(f"Mean centroid stability - Latent dim. {k}:", results["mean_stability"])
        print(f"Std of stability - Latent dim. {k}:", results["std_stability"])

        custom_kmeans_labels = custom_clustering.fit_predict(latents)
        representative_idxs = custom_clustering.get_representatives()
        silh_score = silhouette_score(latents, custom_kmeans_labels, metric=distance_fn)
        cali_score = calinski_harabasz_score(latents, custom_kmeans_labels)
        davies_score = davies_bouldin_score(latents, custom_kmeans_labels)
        metrics = {
            'latent_dim': k,
            'distance_fn': args.distance_fn,
            'mean_stability': results["mean_stability"],
            'std_stability': results["std_stability"],
            'clusters': custom_kmeans_labels,
            'representatives': cids[representative_idxs],
            'silhouette_score': silh_score,
            'calinski_harabasz_score': cali_score,
            'davies_bouldin_score': davies_score
        }
        logs.append(metrics)
        log(INFO, metrics)

    results = pd.DataFrame(logs)
    results.to_csv(f'etc/TimeVAE/{args.location}/results/clusters/latent_dimensions_evaluation_{args.distance_fn}.csv', index_label=False, index=False)

