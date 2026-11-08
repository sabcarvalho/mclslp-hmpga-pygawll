from abc import ABC
from typing import List, Dict, Tuple, Callable
import numpy as np
import gc, random
from src.core.evig import eVIG
from src.core.importance import Importance
from src.core.linkage_learning_mutation import LinkageLearning
from src.config.hardware_config import CACHE_LIMIT
from src.core.fitness_functions import worker_initalize

class GA(ABC):
    EPSILON: float = 1.0e-10
    
    def __init__(
            self,
            *,
            perf_batch: Callable[[List[np.ndarray]], List[float]],
            chrom_size: int,
            mutation_probability: float,
            pop_size: int,
            crossover_rate: float,
            linkage_learning: bool,
            max_time: float,
            params: dict,
            #oriented: bool,
            **kwargs,
    ):
        self.perf_batch: Callable = perf_batch
        self.params: dict = params
        self.chrom_size: int = chrom_size
        self.gurobi_calls: int = 0
        self.oriented = True

        self.stats = {
            "calls": 0,
            "time": 0,
            "generation": -1,
            "fitness_avg_final": 0,
            "infeasible": 0,
            "best_fits":{
                "Fitness": [],
                "Time": [],
                "Calls": [],
                "Generation": []
            },
        }

        # GA hyperparameters
        self.mutation_probability: float = mutation_probability
        self.max_time: float = max_time
        self.population_size: int = pop_size
        self.crossover_rate: float = crossover_rate
        self.linkage_learning: bool = linkage_learning

        # Evolutionary state
        self.evig = eVIG(self.chrom_size, params['n_periods'], params.get("n_families", params["n_items"]))
        self.importance = Importance(self.chrom_size, params['n_periods'], params.get("n_families", params["n_items"]))
        self.mutation_ll = LinkageLearning(self.evig, self.importance,
                                                   self.perf_batch,self._calculate_fitness_value)

        # Performance cache (FIFO strategy)
        self.cache: Dict[bytes, float] = {}

    def _calculate_fitness_value(self, chromosome: np.ndarray, score: float) -> float:
        if score["feasible"]:
            return (0, score["fitness"])
        return (1, score["violation"])

    def _evaluate_individuals(self, individuals) -> None:
        """
        Evaluates a list of individuals using batching and FIFO caching.

        Args:
            individuals: List of Individual objects to be evaluated.
        """
        to_evaluate = []

        # 1. Check cache first to avoid redundant evaluations
        for ind in individuals:
            chrom_hash = tuple(ind.chromosome)
            if chrom_hash in self.cache:
                ind.fitness = self.cache[chrom_hash]
                ind.fitness = self._calculate_fitness_value(ind.chromosome, ind.fitness)
            elif ind.fitness is None:
                to_evaluate.append(ind)

        if not to_evaluate:
            return

        # 2. Perform Batch Evaluation (CPU/GPU Bottleneck)
        chromosomes = [ind.chromosome for ind in to_evaluate]
        self.gurobi_calls += len(chromosomes)
        scores = self.perf_batch(chromosomes)

        # 3. Update Individuals and populate Cache
        for ind, score in zip(to_evaluate, scores):
            ind.fitness = score
            if not score["feasible"]:
                self.stats["infeasible"] +=1
            ind.fitness = self._calculate_fitness_value(ind.chromosome, score)

            # Manage Cache size using FIFO (First-In, First-Out)
            chrom_hash = tuple(ind.chromosome)
            if len(self.cache) >= CACHE_LIMIT:
                # Remove the oldest entry
                self.cache.pop(next(iter(self.cache)))
            self.cache[chrom_hash] = score

    def run(self, seed):
        random.seed(seed)
        worker_initalize(self.params)

    def get_result(self) -> Tuple[np.ndarray, float, float]:
        if not self.final_fittest:
            raise RuntimeError("GAwLL must be executed via .run() before fetching results.")

        return (
            self.final_fittest.chromosome.copy(),
            float(self.final_fittest.fitness[1]),
            float(self.final_fittest.fitness[0]),
            
        )
    def cleanup(self) -> None:
        """
        Releases system resources and breaks circular references.
        Critical for avoiding OOM (Out Of Memory) in long-running experiments.
        """
        self.perf_batch = None
        self.mutation_ll = None
        self.selection = None
        self.crossover = None
        self.mutation = None

        self.cache.clear()
        self.population = []
        self.evig = None
        self.importance = None

        gc.collect()