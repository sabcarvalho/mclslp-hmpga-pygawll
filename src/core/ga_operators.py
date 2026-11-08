"""
Genetic Operators Module.

Provides core mechanisms for evolutionary computation, including 
tournament selection, uniform crossover, and vectorized bit-flip mutation.
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Any


class Selection:
    """Provides selection mechanisms for identifying parents for the next generation."""

    @staticmethod
    def tournament_selection(population: List[Any], tournament_size: int = 3) -> Optional[Any]:
        """
        Selects the fittest individual from a random subset of the population.

        Args:
            population: List of individual objects with a 'fitness' attribute.
            tournament_size: Number of individuals to compete in the tournament.

        Returns:
            The winning individual (highest fitness). Returns None if population is empty.
        """
        if not population:
            return None

        participants = random.sample(population, min(tournament_size, len(population)))


        #minimização-> fallback float('inf') para ignorar invalidos.
        return min(participants, key=lambda ind: ind.fitness if ind.fitness is not None else float('inf'))


class Crossover:
    """Provides genetic recombination operators to create offspring."""

    @staticmethod
    def cxUniform(parent1: List[int], parent2: List[int], indpb: float = 0.5) -> Tuple[List[int],List[int]]:
        """
        Performs uniform crossover using a random binary mask for gene exchange.

        Args:
            parent1: First parent chromosome (binary array).
            parent2: Second parent chromosome (binary array).

        Returns:
            A tuple (offspring1, offspring2) of mutated chromosomes.
        """
        # Generates a boolean mask (50% probability for each gene exchange)
        mask = np.random.random(len(parent1)) < indpb

        # np.where(condition, x, y) returns x if condition is True, else y.
        offspring1 = np.where(mask, parent1, parent2)
        offspring2 = np.where(~mask, parent1, parent2)

        return offspring1.tolist(), offspring2.tolist()
    
    @staticmethod
    def cxSector(ind1: List[int], ind2: List[int], n_fam: int, n_per: int) -> Tuple[List[int], List[int]]:
        off1, off2 = list(ind1), list(ind2)
        rd_per = random.randint(1, n_per - 2)
        rd_fam = random.randint(1, n_fam - 2)

        for i in range(len(ind1)):
            t = i // n_fam
            f = i % n_fam
            if (t > rd_per and f <= rd_fam) or (t <= rd_per and f > rd_fam):
                off1[i] = ind2[i]
            else:
                off2[i] = ind1[i]
                
        return off1, off2

    @staticmethod
    def cxOnePointPeriod(ind1: List[int], ind2: List[int], n_fam: int, n_per: int) -> Tuple[List[int], List[int]]:
        off1, off2 = list(ind1), list(ind2)
        rd_per = random.randint(1, n_per - 2)

        for i in range(len(ind1)):
            t = i // n_fam
            if (t > rd_per):
                off1[i] = ind2[i]
            else:
                off2[i] = ind1[i]
                
        return off1, off2

    @staticmethod    
    def cxOnePointFamily(ind1: List[int], ind2: List[int], n_fam: int) -> Tuple[List[int], List[int]]:
        off1, off2 = list(ind1), list(ind2)
        rd_fam = random.randint(1, n_fam - 2)

        for i in range(len(ind1)):
            f = i % n_fam
            if (f > rd_fam):
                off1[i] = ind2[i]
            else:
                off2[i] = ind1[i]
                
        return off1, off2

    # @staticmethod 
    # def cxUniform(ind1: List[int], ind2: List[int], indpb: float = 0.5) -> Tuple[List[int], List[int]]:
    #     off1, off2 = list(ind1), list(ind2)
    #     size = min(len(off1), len(off2))
    #     for i in range(size):
    #         if random.random() < indpb:
    #             off1[i], off2[i] = off2[i], off1[i]
                
    #     return off1, off2
    

class Mutation:
    """Provides stochastic operators to maintain genetic diversity."""

    @staticmethod
    def bit_flip_mutation(chromosome: List[int], mutation_probability: float) -> List[int]:
        """
        Applies bit-flip mutation by XORing the chromosome with a random mask.

        Args:
            chromosome: The binary chromosome to mutate.
            mutation_probability: Probability (0 to 1) of flipping each bit.

        Returns:
            The mutated chromosome as a boolean/int8 array.
        """
        # Generate a boolean mask where True means 'flip this bit'
        mutation_mask = np.random.random(len(chromosome)) < mutation_probability

        # Logical XOR handles bit flipping (0^0=0, 0^1=1, 1^0=1, 1^1=0)
        # We ensure it returns a boolean array for consistency with Individual class
        return np.logical_xor(chromosome, mutation_mask).astype(bool).tolist()

    @staticmethod
    def mutBitFlipRandom(ind: List[int]) -> List[int]:
        mutant = list(ind)
        rd_indx = random.randint(0, len(mutant) - 1)
        mutant[rd_indx] = not mutant[rd_indx]
        return mutant

    @staticmethod                    
    def mutTwoBitFlipPeriod(ind: List[int], n_fam: int, n_per: int) -> List[int]:
        mutant = list(ind)
        rd_fam = random.randint(0, n_fam - 1)
        rd_per_1 = random.randint(0, n_per - 1)
        rd_per_2 = rd_per_1
        while rd_per_2 == rd_per_1: 
            rd_per_2 = random.randint(0, n_per - 1)

        mutant[rd_per_1 * n_fam + rd_fam] = not mutant[rd_per_1 * n_fam + rd_fam]
        mutant[rd_per_2 * n_fam + rd_fam] = not mutant[rd_per_2 * n_fam + rd_fam]
        
        return mutant

    @staticmethod
    def mutTwoBitFlipFamily(ind: List[int], n_fam: int, n_per: int) -> List[int]:
        mutant = list(ind)
        rd_per = random.randint(0, n_per - 1)
        rd_fam_1 = random.randint(0, n_fam - 1)
        rd_fam_2 = rd_fam_1
        while rd_fam_2 == rd_fam_1: 
            rd_fam_2 = random.randint(0, n_fam - 1)

        mutant[rd_per * n_fam + rd_fam_1] = not mutant[rd_per * n_fam + rd_fam_1]
        mutant[rd_per * n_fam + rd_fam_2] = not mutant[rd_per * n_fam + rd_fam_2]
        
        return mutant