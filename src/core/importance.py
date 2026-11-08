"""
Variable Importance (VImp) Module.

This module tracks the individual contribution of each feature to the model's 
performance. It maintains the empirical performance impact
observed during the evolutionary process.
"""

import numpy as np


class Importance:
    """
    Tracks the empirical importance (VImp) of each variable.

    By storing the historical performance impact for each gene,
    this class provides data for posterior XAI analysis.

    Attributes:
        degree (int): Number of features in the chromosome.
        max_importance (np.ndarray): Vector storing the highest observed impact per gene.
    """

    def __init__(self, degree: int, n_periods: int, n_families: int):
        """
        Initializes the importance tracking vectors using sentinel values.

        Args:
            degree (int): The number of features (chromosome size).
        """
        self.degree: int = degree
        self.n_periods: int = n_periods
        self.n_families: int = n_families
        # Initialized importance vector with zeros
        self._max_importance: np.ndarray = np.zeros(degree, dtype=float)
        self._aggregate_importance_periods: np.ndarray = np.zeros(n_periods, dtype=float)
        self._aggregate_importance_families: np.ndarray = np.zeros(n_families, dtype=float)

    def add_importance(self, i: int, importance: float) -> None:
        """
        Updates the maximum importance for variable 'i'.

        Args:
            i: Index of the variable being updated.
            importance: The measured performance impact.
        """
        if importance > self._max_importance[i]:
            t = i // self.n_families
            f = i % self.n_families

            self._aggregate_importance_families[f] -= self._max_importance[i]
            self._aggregate_importance_periods[t] -= self._max_importance[i]

            self._max_importance[i] = importance

            self._aggregate_importance_families[f] += importance
            self._aggregate_importance_periods[t] += importance

    def export_importance_vector(self) -> np.ndarray:
        """
        Returns the VImp vector.

        Returns:
            np.ndarray: Max_Importance
        """
        return self._max_importance.copy()

    def export_importance_families(self):
        return self._aggregate_importance_families
    def export_importance_periods(self):
        return self._aggregate_importance_periods
    def export_biggest_imp_family(self):
        return np.unravel_index(np.argmax(self._aggregate_importance_families, self._aggregate_importance_families.shape))
    def export_biggest_imp_period(self):
        return np.unravel_index(np.argmax(self._aggregate_importance_periods, self._aggregate_importance_periods.shape))