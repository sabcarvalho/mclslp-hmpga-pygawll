import random, time, gc
import numpy as np

from src.core.fitness_functions import solve_model_parallel, fix_and_optimize
from core.population_hmpga import TernaryTreePopulation, Node
from src.core.ga_operators import Crossover, Mutation
from src.core.ga import GA

def clone_individual(ind):
    new_ind = Node(ind.chromosome.copy(), ind.fitness)
    return new_ind

class Hmpga(GA):
    EPSILON: float = 1.0e-10
    def __init__(self,
                n_trees: int,
                **kwargs,
        ): 
        self.n_trees: int = n_trees
        super().__init__(**kwargs)

        self._create_model()

    def _create_model(self):
        self.mutations = [
            lambda ind: Mutation.mutBitFlipRandom(ind.chromosome),
            lambda ind: Mutation.mutTwoBitFlipPeriod(ind.chromosome, n_fam=self.params["n_families"], n_per=self.params["n_periods"]),
            lambda ind: Mutation.mutTwoBitFlipFamily(ind.chromosome, n_fam=self.params["n_families"], n_per=self.params["n_periods"])
        ]
        
        self.crossovers = [
            lambda p1, p2: Crossover.cxUniform(p1.chromosome, p2.chromosome, indpb=0.5),
            lambda p1, p2: Crossover.cxSector(p1.chromosome, p2.chromosome, n_fam=self.params["n_families"], n_per=self.params["n_periods"]),
            lambda p1, p2: Crossover.cxOnePointPeriod(p1.chromosome, p2.chromosome, n_fam=self.params["n_families"], n_per=self.params["n_periods"]),
            lambda p1, p2: Crossover.cxOnePointFamily(p1.chromosome, p2.chromosome, n_fam=self.params["n_families"])
        ]

    def _create_heuristic_individual(self):
        ind = Node([1] * self.chrom_size, None)
        
        result, _, lot_sizes = solve_model_parallel(ind.chromosome, return_lot_sizes=True)
        
        if result["feasible"]:
            n_fam = self.params["n_families"]
            n_per = self.params["n_periods"]
            
            for t in range(n_per):
                for f in range(n_fam):
                    producted_fam = sum(
                        lot_sizes[t, j] for j in range(self.params["n_items"]) 
                        if self.params["item_family"].get(j) == f
                    )
                    
                    if producted_fam == 0:
                        idx = t * n_fam + f
                        ind.chromosome[idx] = 0
        else:
            ind = Node([random.randint(0,1) for _ in range(self.chrom_size)], None)
        
        return ind
    
    def run(self, seed, n_migrants=1):
        super().run(seed)
        nCrossovers = self.population_size * self.crossover_rate
        best_fitness_track = float('inf')
        
        init_train = time.perf_counter()

        self.trees = [TernaryTreePopulation() for _ in range(self.n_trees)]

        for i in range(self.n_trees):
            pop_inicial = [self._create_heuristic_individual() for _ in range(self.population_size)]
            self._evaluate_individuals(pop_inicial)
            for ind in pop_inicial:
                self.trees[i].insert_node(ind)
        p_window = 2
        f_window = 2
        while((time.perf_counter() - init_train) < self.max_time):  
            for i in range(3):
                converged = False
                gens = 0
                while(not converged):
                    gens+=1
                    offspring = []
                    for _ in range(int(nCrossovers)):
                        p1, p2 = self.trees[i].random_leader_supporter()
                        p1 = clone_individual(p1)
                        p2 = clone_individual(p2)
                        p1.fitness = None
                        p2.fitness = None
                        if random.random() < self.crossover_rate:
                            cx = random.randint(0, len(self.crossovers)-1)
                            p1.chromosome, _ = self.crossovers[cx](p1, p2)
                        if random.random() < self.mutation_probability:
                            mut = random.randint(0, len(self.mutations))
                            if mut == len(self.mutations):
                                p1.chromosome = self.mutations[0](p1)
                                p1.chromosome = self.mutations[0](p1)
                            else:
                                p1.chromosome = self.mutations[mut](p1)

                        offspring.append(p1)

                    if self.linkage_learning:
                        for _ in range(len(offspring), self.population_size - 2):
                            p, _ = self.trees[i].random_leader_supporter()
                            p = clone_individual(p)
                            chroms = self.mutation_ll.mutation_ll(p)
                            offspring.extend([Node(ind.copy(), None) for ind in chroms])
                    
                    self._evaluate_individuals(offspring)
                    converged = True
                    for ind in offspring:
                        inserted = self.trees[i].insert_node(ind)
                        converged = converged and not inserted

            best_individual = min([self.trees[i].root for i in range(self.n_trees)], key= lambda x: x.fitness)
            current_fit = best_individual.fitness
            best_chrm = clone_individual(best_individual)

            migrants_01 = self.trees[0].best_nodes(n_migrants)
            migrants_12 = self.trees[1].best_nodes(n_migrants)
            migrants_20 = self.trees[2].best_nodes(n_migrants)

            for ind in migrants_01: 
                self.trees[1].insert_node(clone_individual(ind))
            for ind in migrants_12: 
                self.trees[2].insert_node(clone_individual(ind))
            for ind in migrants_20: 
                self.trees[0].insert_node(clone_individual(ind))
            if ((time.perf_counter() - init_train) >= self.max_time): break

            copy_node = clone_individual(best_individual)
            
            fix_and_optimize(copy_node, period_window_size=p_window, family_window_size=f_window,
                             init_train=init_train, max_time=self.max_time)
            
            if current_fit > copy_node.fitness:
                best_chrm.fitness = copy_node.fitness
                best_chrm.chromosome = copy_node.chromosome.copy()
                for i in range(self.n_trees):
                    self.trees[i].insert_node(clone_individual(best_chrm))
            else:
                p_window += 1
                f_window += 1
            
            if best_chrm.fitness[0] == 0 and best_fitness_track - best_chrm.fitness[1] > self.EPSILON:
                best_fitness_track = best_chrm.fitness[1]
                time_optim = time.perf_counter() - init_train

                self.stats["best_fits"]["Fitness"].append(best_fitness_track)
                self.stats["best_fits"]["Time"].append(time_optim)
                self.stats["best_fits"]["Calls"].append(self.gurobi_calls)
                self.stats["best_fits"]["Generation"].append(0)
                

        final_fitness = []
        for i in range(self.n_trees):
            nodes = self.trees[i].get_all_nodes()
            fits = [node.fitness[1] for node in nodes]
            final_fitness += fits

        final_fitness = np.array(final_fitness)
        self.stats["fitness_avg_final"] = np.mean(final_fitness)
        self.final_fittest = min([self.trees[i].root for i in range(self.n_trees)], key= lambda x: x.fitness)

    def cleanup(self):
        self.final_fittest = None
        self.cache.clear()
        gc.collect()