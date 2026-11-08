"""
GAwLL: Genetic Algorithm with Linkage Learning Engine.

This module implements the core evolutionary logic, integrating parsimony
pressure, stagnation-based resets (Tau-Reset), and Linkage Learning
capabilities for high-dimensional feature selection.
"""

import gc, time
import random
import numpy as np
from typing import List, Optional, Tuple

# Project-specific imports
from src.core.evig import eVIG
from src.core.importance import Importance
from src.utils.statistic import PopulationAnalyzer
from src.core.ga_operators import Selection, Crossover, Mutation
from src.core.fitness_functions import fix_and_optimize
from src.core.ga import GA


class Individual:
    """
    Represents a single solution in the evolutionary population.

    Using __slots__ optimizes memory by preventing the creation of __dict__
    for each instance, which is critical for GA performance in Python.

    Attributes:
        chromosome (np.ndarray): Binary vector representing selected features.
        fitness (Optional[float]): Scalar value (Performance + Parsimony).
        f_perf (Optional[float]): Raw performance metric (e.g., accuracy).
    """
    __slots__ = ['chromosome', 'fitness', 'f_perf']

    def __init__(
            self,
            chromosome: List[int],
            fitness: Optional[float] = None,
            f_perf: Optional[float] = None
    ):
        self.chromosome: List[int] = chromosome
        self.fitness: Optional[float] = fitness
        self.f_perf: Optional[float] = f_perf


class GAwLL(GA):
    """
    Main Engine for the Genetic Algorithm with Linkage Learning (GAwLL).

    This class orchestrates the evolutionary lifecycle: initialization,
    evaluation, selection, recombination, and mutation.

    Attributes:
        EPSILON (float): Small constant for robust floating-point comparisons.
    """
    EPSILON: float = 1.0e-10

    def __init__(
            self,
            max_generations: int,
            tau_reset: int,
            freq_fix: int,
            elite_rate: float,
            random_rate: float,
            **kwargs
    ):
        """
        Initializes the GAwLL instance with evolutionary hyperparameters.

        Args:
            perf_batch: Function to evaluate a batch of chromosomes.
            chrom_size: Length of the binary chromosome.
            mutation_probability: Rate of bit-flip mutation.
            max_generations: Maximum number of generations to run.
            pop_size: Size of the population.
            crossover_rate: Probability of crossover occurrence.
            tau_reset: Generations of stagnation before triggering a reset.
            linkage_learning: Boolean flag to enable Linkage Learning mutation.
        """
        self.max_generations: int = max_generations
        self.tau_reset_generations: int = tau_reset
        self.freq_fix = freq_fix# 5
        self.elite_rate = elite_rate#0.05
        self.random_rate = random_rate#0.01
        super().__init__(**kwargs)

        self.crossover: Optional[Crossover] = None
        self.mutation: Optional[Mutation] = None
        self.population_analyzer: Optional[PopulationAnalyzer] = None
        self.selection: Optional[Selection] = None

    def _initialize_population(self) -> None:
        """Initializes the population with random binary chromosomes."""
        #bits = np.random.randint(0, 2, size=(self.population_size, self.chrom_size)).astype(bool)
        
        self.population = [Individual([random.randint(0,1) for _ in range(self.chrom_size)]) for _ in range(self.population_size)]
        self._evaluate_individuals(self.population)

    def _apply_elitism(self, new_population: List[Individual], fittest_individual: Individual) -> None:
        """
        Protects the best found solution by injecting it into the new generation.

        Args:
            new_population: The generation currently being formed.
            fittest_individual: The best individual from the previous state.
        """

        worst_idx = max(range(len(new_population)), key=lambda i: new_population[i].fitness if new_population[i].fitness is not None else float('inf'))

        # Injeta uma cópia do melhor indivíduo
        new_population[worst_idx] = Individual(
            fittest_individual.chromosome.copy(),
            fittest_individual.fitness,
            fittest_individual.f_perf
        )

    def _generation(self, fittest_individual: Individual) -> List[Individual]:
        """
        Performs a standard evolutionary generation cycle.

        Args:
            fittest_individual: The current global best for elitism.

        Returns:
            List[Individual]: The newly formed population.
        """
        new_population: List[Individual] = []

        while len(new_population) < self.population_size:
            parent1 = self.selection.tournament_selection(self.population)

            if len(new_population) < self.population_size - 1:
                parent2 = self.selection.tournament_selection(self.population)

                off1_chrom, off2_chrom = parent1.chromosome.copy(), parent2.chromosome.copy()

                if random.random() < self.crossover_rate:
                    off1_chrom, off2_chrom = self.crossover.cxUniform(off1_chrom, off2_chrom)

                self.mutation.bit_flip_mutation(off1_chrom, self.mutation_probability)
                self.mutation.bit_flip_mutation(off2_chrom, self.mutation_probability)

                new_population.extend([Individual(off1_chrom), Individual(off2_chrom)])
            else:
                off_chrom = parent1.chromosome.copy()
                self.mutation.bit_flip_mutation(off_chrom, self.mutation_probability)
                new_population.append(Individual(off_chrom))

        self._evaluate_individuals(new_population)
        self._apply_elitism(new_population, fittest_individual)
        return new_population

    def _generation_ll(self, fittest_individual: Individual) -> List[Individual]:
        """
        Evolutionary generation cycle enhanced with Linkage Learning mutation.

        Args:
            fittest_individual: The current global best for elitism.

        Returns:
            List[Individual]: The newly formed population.
        """
        new_population: List[Individual] = []

        while len(new_population) < self.population_size:
            parent1 = self.selection.tournament_selection(self.population)

            # 1. Exploratory Crossover
            num_crossover = int(self.population_size * self.crossover_rate)
            if len(new_population) < num_crossover:
                parent2 = self.selection.tournament_selection(self.population)
                off1_chrom, off2_chrom = self.crossover.cxUniform(
                    parent1.chromosome.copy(), parent2.chromosome.copy()
                )
                self.mutation.bit_flip_mutation(off1_chrom, self.mutation_probability)
                self.mutation.bit_flip_mutation(off2_chrom, self.mutation_probability)
                new_population.extend([Individual(off1_chrom), Individual(off2_chrom)])

            # 2. Linkage Learning Guided Mutation
            elif len(new_population) < self.population_size - 2:
                chroms = self.mutation_ll.mutation_ll(parent1)
                new_population.extend([Individual(c) for c in chroms])

            # 3. Residual filling
            else:
                off_chrom = parent1.chromosome.copy()
                self.mutation.bit_flip_mutation(off_chrom, self.mutation_probability)
                new_population.append(Individual(off_chrom))

        self._evaluate_individuals(new_population)
        self._apply_elitism(new_population, fittest_individual)
        return new_population

    def run(self, seed: int) -> None:
        """
        Executes the full GAwLL evolutionary process.

        Args:
            seed: Random seed for reproducibility.
        """
        super().run(seed)
        np.random.seed(seed)
        
        init_train = time.perf_counter()
        # Component Initialization
        self._initialize_population()
        
        self.population_analyzer = PopulationAnalyzer()
        self.selection = Selection()
        self.crossover = Crossover()
        self.mutation = Mutation()
        
        last_improvement_gen = 0
        best_fitness_track = float('inf')

        generation = 1
        while((time.perf_counter() - init_train) < self.max_time):
        #for generation in range(self.max_generations):
            fittest = self.population_analyzer.get_fittest_individual(self.population)

            # Stagnation tracking
            if fittest.fitness[0] == 0 and best_fitness_track - fittest.fitness[1] > self.EPSILON:
                best_fitness_track = fittest.fitness[1]
                last_improvement_gen = generation
                time_optim = time.perf_counter() - init_train

                self.stats["best_fits"]["Fitness"].append(best_fitness_track)
                self.stats["best_fits"]["Time"].append(time_optim)
                self.stats["best_fits"]["Calls"].append(self.gurobi_calls)
                self.stats["best_fits"]["Generation"].append(generation)

            # Tau-Reset logic: Triggered if no improvement is found
            if generation - last_improvement_gen > self.tau_reset_generations:
                print(f"    [RESTART] Generation {generation}: Perturbing population...")
                reset_pop = []
                for _ in range(self.population_size):
                    c = fittest.chromosome.copy()
                    self.mutation.bit_flip_mutation(c, 0.30)  # High mutation for reset
                    reset_pop.append(Individual(c))

                self._evaluate_individuals(reset_pop)
                self.population = reset_pop
                last_improvement_gen = generation

            # Evolution Step
            if self.linkage_learning:
                self.population = self._generation_ll(fittest)
            else:
                self.population = self._generation(fittest)

            if generation > 0 and (generation % self.freq_fix == 0 ):
                self.population.sort(key=lambda ind: ind.fitness[1])
                
                num_elite = max(0, int(self.population_size * self.elite_rate))
                num_random = max(0, int(self.population_size * self.random_rate))
                
                for i in range(num_elite):
                    fix_and_optimize(self.population[i],init_train=init_train, max_time=self.max_time)
                    
                num_random = random.sample(self.population[num_elite:], num_random)
                for ind in num_random:
                    fix_and_optimize(ind,init_train=init_train, max_time=self.max_time)
            
            #print(f"Generation {generation} finished")
            generation+=1
            gc.collect()

        fittest = self.population_analyzer.get_fittest_individual(self.population)
        
        # Stagnation tracking
        if fittest.fitness[0] == 0 and best_fitness_track - fittest.fitness[1] > self.EPSILON:
            best_fitness_track = fittest.fitness[1]
            last_improvement_gen = generation
            time_optim = time.perf_counter() - init_train

            self.stats["best_fits"]["Fitness"].append(best_fitness_track)
            self.stats["best_fits"]["Time"].append(time_optim)
            self.stats["best_fits"]["Calls"].append(self.gurobi_calls)
            self.stats["best_fits"]["Generation"].append(generation)
        
        if self.population:
            final_fitness = []
            for i in range(self.population_size):
                final_fitness.append(self.population[i].fitness[1])
            
            final_fitness = np.array(final_fitness)
            self.stats["fitness_avg_final"] = np.mean(final_fitness)
            self.final_fittest = self.population_analyzer.get_fittest_individual(self.population)

    def get_feature_selection_results(self) -> Tuple[np.ndarray, float, float]:
        """
        Extracts the results from the best individual found.

        Returns:
            Tuple[np.ndarray, float, float]: (Best Chromosome, Fitness, Raw Performance)
        """
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