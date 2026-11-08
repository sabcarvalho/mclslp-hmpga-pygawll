"""
Empirical Variable Interaction Graph (eVIG) Module.

This module implements the storage and management of the Interaction Matrix (VInt).
It captures empirical evidence of relationships between features by recording
performance deltas observed during the Linkage Learning mutation process.
"""

import numpy as np


class eVIG:
    """
    Manages the Interaction Matrix (VInt) for feature linkage discovery.

    The eVIG maintains symmetric adjacency matrices that store the
    interaction weights observed between pairs of variables.

    Attributes:
        degree (int): The number of features (dimension of the square matrices).
    """

    def __init__(self, degree: int, n_periods: int, n_families: int):
        """
        Initializes the interaction matrices with zeros.

        Args:
            degree (int): Total number of variables/features in the dataset.
        """
        self.degree: int = degree
        self.n_periods: int = n_periods
        self.n_families: int = n_families

        # _max_edge_weight: Stores the highest interaction.
        self._max_edge_weight: np.ndarray = np.zeros((degree, degree), dtype=float)
        self._aggregate_interaction: np.ndarray = np.zeros((n_periods, n_families), dtype=float)
        self._aggregate_interaction_families: np.ndarray = np.zeros((n_families, n_families), dtype=float)
        self._aggregate_interaction_periods: np.ndarray = np.zeros((n_periods, n_periods), dtype=float)

    def add_edge(self, a: int, b: int, w: float) -> None:
        """
        Records an interaction weight between feature 'a' and feature 'b'.

        This method updates the symmetric matrices if the new weight 'w'
        represents a new maximum for the pair.

        Args:
            a (int): Index of the first feature.
            b (int): Index of the second feature.
            w (float): The interaction weight.
        """
        # Symmetric update for the Maximum Interaction Matrix
        if w > self._max_edge_weight[a,b]:
            t_a = a // self.n_families
            f_a = a % self.n_families

            t_b = b // self.n_families
            f_b = b % self.n_families

            self._aggregate_interaction[t_a, f_b] -= self._max_edge_weight[a, b]
            self._aggregate_interaction[t_b, f_a] -= self._max_edge_weight[a, b]

            self._aggregate_interaction_families[f_a, f_b] -= self._max_edge_weight[a, b]
            self._aggregate_interaction_periods[t_a, t_b] -= self._max_edge_weight[a, b]

            self._max_edge_weight[a, b] = w
            self._max_edge_weight[b, a] = w

            self._aggregate_interaction[t_a, f_b] += w
            self._aggregate_interaction[t_b, f_a] += w

            self._aggregate_interaction_families[f_a, f_b] += w
            self._aggregate_interaction_periods[t_a, t_b] += w

    def export_interaction_matrix(self) -> np.ndarray:
        """
        Returns a copy of the internal Maximum Variable Interaction Matrix (vInt Max).

        This matrix represents the peak interaction strengths detected by
        the Linkage Learning mechanism during the evolution.

        Returns:
            np.ndarray: A 2D symmetric matrix of shape (n_features, n_features).
        """
        return self._max_edge_weight.copy()

    def export_interaction_families(self): 
        return self._aggregate_interaction_families
    def export_interaction_periods(self): 
        return self._aggregate_interaction_periods
    def export_biggest_int_family(self):
        return np.unravel_index(np.argmax(self._aggregate_interaction_families), self._aggregate_interaction_families.shape)
    def export_biggest_int_period(self):
        return np.unravel_index(np.argmax(self._aggregate_interaction_periods), self._aggregate_interaction_periods.shape)