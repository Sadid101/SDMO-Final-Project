from collections import defaultdict
import math
from abc import ABC, abstractmethod
from typing import Callable

__all__ = ['BaseTemporalKatzCentrality', 'TemporalKatzCentrality',
           'TruncatedTemporalKatzCentrality', 'LazyTemporalKatzCentrality']


class BaseTemporalKatzCentrality(ABC):
    """Abstract Base Class for Temporal Katz Centrality.

    Can be used to define own versions of Katz Centrality for specific decays or different
    data storing mechanisms.
    Subclasses must implement update(u,v,t) for a new edge (u → v) at time t
    and query(u, t) for score of u at time t.

    """
    @abstractmethod
    def update(self, u, v, t) -> float:
        """Update the graph with edge (u → v) at time t using general update rule.

        Parameters
        ----------
        u: Any
            source node of the edge
        v: Any
            target node of the edge
        t: int
            timestamp of the edge

        Returns
        -------
        float:
            r_t[v], the new score of node v at time t
        """
        pass

    @abstractmethod
    def query(self, u, t) -> float:
        """Compute the centrality of node u at time t.

        Parameters
        ----------
        u: Any
            node
        t: int
            timestamp to decay paths to

        Returns
        -------
        float:
        r_t[u], the new score of node u at time t
        """
        pass


class TemporalKatzCentrality(BaseTemporalKatzCentrality):
    """Implementation of TemporalKatzCentrality for a general case (Eq. 10, 11).
    Béres, F., Pálovics, R., Oláh, A., & Benczúr, A. A. (2018).
    Temporal walk based centrality metric for graph streams.
    Applied network science, 3(1), 32.
    """
    def __init__(self, decay_function: Callable[[float], float]) -> None:
        """Initialize the Temporal Katz Centrality with the given decay function.

        Parameters
        ----------
        decay_function:
            function float -> float to be used for path decay
        """
        self.phi = decay_function
        self.edge_data = defaultdict(list)  # v → list of (z, t_zv, w_zv)

    def update(self, u, v, t) -> float:
        """Update the graph with edge (u → v) at time t using general update rule.

        Parameters
        ----------
        u: Any
            source node of the edge
        v: Any
            target node of the edge
        t: int
            timestamp of the edge

        Returns
        -------
        float:
            r_t[v], the new score of node v at time t
        """

        # Compute r_u(t)
        r_u = sum(w_zu * self.phi(t - t_zu) for z, t_zu, w_zu in self.edge_data[u])

        # Add new edge (u → v) with weight = 1 + r_u
        self.edge_data[v].append((u, t, (1 + r_u)))

        return self.query(v, t)

    def query(self, u, t) -> float:
        """Compute the centrality of node u at time t.

        Parameters
        ----------
        u: Any
            node
        t: int
            timestamp to decay paths to

        Returns
        -------
        float:
            r_t[u], the new score of node u at time t
        """
        return sum(
            w_zv * self.phi(t - t_zv)
            for z, t_zv, w_zv in self.edge_data[u]
            if t_zv < t
        )


class TruncatedTemporalKatzCentrality(BaseTemporalKatzCentrality):
    """Implementation of TemporalKatzCentrality truncating length of walks (Eq. 12, 13).
    Béres, F., Pálovics, R., Oláh, A., & Benczúr, A. A. (2018).
    Temporal walk based centrality metric for graph streams.
    Applied network science, 3(1), 32.
    """
    def __init__(self, decay_function: Callable[[float], float], K: int):
        """Initialize the Temporal Katz Centrality with the given decay function and max walk length K.

        Parameters
        ----------
        decay_function:
            function float -> float to be used for path decay
        K: int
            maximum walk length to consider
        """
        self.phi = decay_function
        self.K = K
        # edge_data[k][v] = list of (z, t_zv, w_zv) for paths of length k ending at v
        self.edge_data = [defaultdict(list) for _ in range(K + 1)]  # index 0 unused

    def update(self, u, v, t) -> float:
        """Update the graph with edge (u → v) at time t using general update rule.

        Parameters
        ----------
        u: Any
            source node of the edge
        v: Any
            target node of the edge
        t: int
            timestamp of the edge

        Returns
        -------
        float:
            r_t[v], the new score of node v at time t
        """
        self.edge_data[1][v].append((u, t, 1.0))
        for k in range(2, self.K + 1):
            score = sum(
                w_zu * self.phi(t - t_zu)
                for z, t_zu, w_zu in self.edge_data[k - 1][u]
                if t_zu < t
            )
            if score > 0:
                self.edge_data[k][v].append((u, t, score))
        return self.query(v, t)

    def query(self, u, t) -> float:
        """Compute the centrality of node u at time t.

        Parameters
        ----------
        u: Any
            node
        t: int
            timestamp to decay paths to

        Returns
        -------
        float:
            r_t[u], the new score of node u at time t
        """
        r = 0.0
        for k in range(1, self.K + 1):
            for z, t_zv, w_zv in self.edge_data[k][u]:
                if t_zv < t:
                    r += w_zv * self.phi(t - t_zv)
        return r


class LazyTemporalKatzCentrality(BaseTemporalKatzCentrality):
    """Implementation of TemporalKatzCentrality for exponential decay with lazy decay (Eq. 14, 15).
    Béres, F., Pálovics, R., Oláh, A., & Benczúr, A. A. (2018).
    Temporal walk based centrality metric for graph streams.
    Applied network science, 3(1), 32.
    """
    def __init__(self, beta: float, c: float):
        """Initialize Lazy Katz centrality with parameters of exponential decay

        Parameters
        ----------
        beta: float
            scaling parameter of β·exp(−ct)
        c: float
            decay parameter of β·exp(−ct)
        """
        self.beta = beta
        self.c = c
        self.r = defaultdict(float)
        self.t_last = defaultdict(lambda: -math.inf)  # last updated time for each node

    def _decay(self, delta_t: float) -> float:
        """Decay multiplier for elapsed time."""
        return math.exp(-self.c * delta_t)

    def update(self, u, v, t) -> float:
        """Update the graph with edge (u → v) at time t using the lazy update rule for exponential decay.

        Parameters
        ----------
        u: Any
            source node of the edge
        v: Any
            target node of the edge
        t: int
            timestamp of the edge

        Returns
        -------
        float:
            r_t[v], the new score of node v at time t
        """

        # Apply lazy decay
        self.r[u] = self.r[u] * self._decay(t - self.t_last[u])
        self.r[v] = self.r[v] * self._decay(t - self.t_last[v]) + (1 + self.r[u]) * self.beta

        # Update timestamps
        self.t_last[u] = t
        self.t_last[v] = t

        return self.r[v]

    def query(self, u, t) -> float:
        """Compute the centrality of node u at time t. Only works for t > t_last[u].

        Parameters
        ----------
        u: Any
            node
        t: int
            timestamp to decay paths to

        Returns
        -------
        float:
            r_t[u], the new score of node u at time t

        Raises
        ------
        ValueError:
            If t is less than t_last[u], i.e., the most recent time node u had an edge
        """
        if t < self.t_last[u]:
            raise ValueError(f"Can only compute centrality for node {u} for t > {self.t_last[u]}, got {t}")
        return self.r[u] * self._decay(t - self.t_last[u])