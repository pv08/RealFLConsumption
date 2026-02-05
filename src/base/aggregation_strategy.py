from typing import *
import numpy as np
from src.utils.functions import get_params
from src.utils.aggregation_functions import *

class Aggregator:
    def __init__(self, aggregation_alg, params: Optional[dict]=None):
        self.alg = aggregation_alg
        if params is None:
            params = get_params(aggregation_alg)

        self.params = params
        self.v_t = None
        self.m_t = None
        self.momentum_vector = None

    def aggregate(self, local_weights: List[Tuple[List[np.ndarray], int]], current_model: List[np.ndarray]) -> List[np.ndarray]:
        aggregated = []
        if self.alg == "fedavg" or self.alg == "fedprox":
            aggregated = fedavg_aggregate(local_weights)
        elif self.alg == "avg":
            aggregated = simple_aggregate(local_weights)
        elif self.alg == "medianavg":
            aggregated = median_aggregate(local_weights)
        elif self.alg == "fednova_aggregate":
            aggregated = fednova_aggregate(local_weights, current_model, rho=self.rho)
        elif self.alg == "fedadagrad":
            aggregated, m_t, v_t = fedadagrad_aggregate(local_weights, current_model,
                                                        m_t=self.m_t, v_t=self.v_t, beta_1=self.beta_1, eta=self.eta, tau=self.tau)
            self.m_t, self.v_t = m_t, v_t
        elif self.alg == "fedyogi":
            aggregated, m_t, v_t = fedyogi_aggregate(local_weights, current_model,
                                                     m_t=self.m_t, v_t=self.v_t, beta_1=self.beta_1, beta_2=self.beta_2,
                                                     eta=self.eta, tau=self.tau)
            self.m_t, self.v_t = m_t, v_t
        elif self.alg == "fedadam":
            aggregated, m_t, v_t = fedadam_aggregate(local_weights, current_model,
                                                     m_t=self.m_t, v_t=self.v_t, beta_1=self.beta_1, beta_2=self.beta_2,
                                                     eta=self.eta, tau=self.tau)
            self.m_t, self.v_t = m_t, v_t
        elif self.alg == "fedavgm":
            aggregated, momentum_vector = fedavgm_aggregate(local_weights, current_model, server_momentum=self.server_momentum,
                                                     momentum_vector=self.momentum_vector, server_lr=self.server_lr)
            self.momentum_vector = momentum_vector
        return aggregated

    def __repr__(self):
        if self.alg == "fedavg":
            rep = f"FedAvg()"
        elif self.alg == "avg":
            rep = f"SimpleAvg()"
        elif self.alg == "medianavg":
            rep = f"MedianAvg()"
        elif self.alg == "fedprox":
            self.mu = self.params["mu"]
            rep = f"FedProx(mu={self.mu})"
        elif self.alg == "fednova":
            self.rho = self.params["rho"]
            rep = f"FedNova(rho={self.rho})"
        elif self.alg == "fedadagrad":
            self.beta_1 = self.params["beta_1"]
            self.eta = self.params["eta"]
            self.tau = self.params["tau"]
            rep = f"FedAdagrad(beta_1={self.beta_1}, eta={self.eta}, tau={self.tau})"
        elif self.alg == "fedyogi":
            self.beta_1 = self.params["beta_1"]
            self.beta_2 = self.params["beta_2"]
            self.eta = self.params["eta"]
            self.tau = self.params["tau"]
            rep = f"FedYogi(beta_1={self.beta_1}, beta_2={self.beta_2}, eta={self.eta}, tau={self.tau})"
        elif self.alg == "fedadam":
            self.beta_1 = self.params["beta_1"]
            self.beta_2 = self.params["beta_2"]
            self.eta = self.params["eta"]
            self.tau = self.params["tau"]
            rep = f"FedAdam(beta_1={self.beta_1}, beta_2={self.beta_2}, eta={self.eta}, tau={self.tau})"
        elif self.alg == "fedavgm":
            self.server_momentum = self.params["server_momentum"]
            self.server_lr = self.params["server_lr"]
            rep = f"FedAvgM(server_momentum={self.server_momentum}, server_lr={self.server_lr})"
        else:
            raise NotImplementedError
        return rep
