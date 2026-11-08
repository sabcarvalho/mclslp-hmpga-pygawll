import numpy as np
import os
import pandas as pd
from pathlib import Path

class Statistics:
    """
    Manages the evolutionary data, XAI metrics, and statistical analysis across runs.
    Centralizes result persistence in text reports, visualizations, and SQL.
    """

    def __init__(self, plotter, grapher, saver, db_logger):
        """
        Initializes the Statistics manager.

        Args:
            plotter: Object for generating importance plots.
            grapher: Object for interaction matrices visualizations.
            saver: Object for file system operations (CSV, TXT).
            db_logger: Object for SQL database persistence.
        """
        self.plotter = plotter
        self.grapher = grapher
        self.saver = saver
        self.db_logger = db_logger

        # Directory handling - uses saver's path if available
        
        self.output_path = getattr(saver, 'output_path', Path('results'))
        self.output_path.mkdir(parents=True, exist_ok=True)

    def save_all_results(self, best_ind_run, aggregates, config, stats):
        """
        Main entry point to save all experiment outputs after completion.
        Note: Execution times are handled via SQL during the runs.

        Args:
            best_ind_run (list): Results per run (chromosome, fitness, performance).
            aggregates (dict): Calculated means for VInt, VImp and comparative methods.
            config (dict): Experiment configuration parameters.
        """
        # 1. Save XAI Metrics (Matrices and Plots)
        self.save_xai_metrics(config, **aggregates)

        # 2. Save Feature Selection Statistics (Text Report)
        self.save_feature_selection_statistics(best_ind_run, config, stats)

        # 3. Finalize SQL Summary (Aggregates performance in DB)
        self.db_logger.finalize_summary(aggregates['id_group'])

    def save_xai_metrics(self, config, vint_gawll, vimp_gawll, vimp_perm, vimp_intr, **kwargs):
        """Saves interaction matrices, importance vectors, and generates plots."""
        prefix = f"{config['dataset_name']}_{config['model_label']}"
        vars_names = list(config['variables'])

        # CSV Data (Raw matrices for future analysis)
        self.saver.save_interaction_matrix(vint_gawll, f"{prefix}_Mean_VInt_GAwLL")
        self.saver.save_importances(vimp_gawll, f"{prefix}_Mean_VImp_GAwLL")

        if vimp_perm is not None: self.saver.save_importances(vimp_perm, f"{prefix}_Mean_VImp_Permutation")
        if vimp_intr is not None: self.saver.save_importances(vimp_intr, f"{prefix}_Mean_VImp_Intrinsic")

        # Top Rankings (Detailed TXT for quick inspection)
        self.saver.save_top_importances(vimp_gawll, vars_names, config['model_label'], config['dataset_name'],
                                        config['top_n'])
        self.saver.save_top_interactions(vint_gawll, vars_names, config['model_label'], config['dataset_name'],
                                         config['top_n'])

        # Visualizations (PNG/PDF)
        self.grapher.save_all_visualizations(vimp_gawll, vint_gawll, vars_names, prefix)
        self.plotter.generate_importance_plots(vars_names, config['model_label'],
                                               config['dataset_name'], vimp_gawll)

    def save_feature_selection_statistics(self, best_ind_run, config, stats):
        """
        Generates a consolidated text report with detailed statistics on fitness,
        accuracy, and feature subset sizes.

        Args:
            best_ind_run (list): Results from each run (individual, fitness, performance).
            config (dict): Experiment configuration including variables names and model label.
        """
        
        vars_names = list(config['variables'])

        # Extraction of metrics and feature counts per run
        all_fitness = np.array([ind[1] for ind in best_ind_run])
        all_counts = np.array([int(np.sum(ind[0])) for ind in best_ind_run])

        # Identifying critical indices for extreme cases
        idx_max_fit = np.argmax(all_fitness)
        idx_min_fit = np.argmin(all_fitness)

        # Global best solution based on Max Fitness
        best_chrom = best_ind_run[idx_max_fit][0]
        selected_vars = [vars_names[i] for i, val in enumerate(best_chrom) if val == 1]

        report = [
            "=" * 75,
            f"CONSOLIDATED REPORT: {config['dataset_name'].upper()} | {config['model_label'].upper()}",
            "=" * 75 + "\n",
            f"Time: {np.mean(stats['exec_time'])} +/- {np.std(stats['exec_time'])} \n",
            "1. RESULTS PER RUN",
            "-" * 35
        ]

        for i, (chrom, fit, perf) in enumerate(best_ind_run):
            report.append(f"Run {i:02d} | Fitness: {fit:.5f} | Acc: {perf:.5f} | Vars: {all_counts[i]}")

        report.extend([
            "\n2. GLOBAL BEST SOLUTION (MIN Fitness)",
            "-" * 35,
            f"Best Fitness:      {all_fitness[idx_min_fit]:.5f}",
            f"Features Selected: {all_counts[idx_min_fit]}",
            f"Names: {', '.join(selected_vars)}",

            "\n3. EXTREME METRICS",
            "-" * 35,
            f"Max Fitness:  {all_fitness[idx_max_fit]:.5f} (Vars: {all_counts[idx_max_fit]})",
            f"Min Fitness:  {all_fitness[idx_min_fit]:.5f} (Vars: {all_counts[idx_min_fit]})",

            "\n4. GENERAL STATISTICS",
            "-" * 35,
            f"Fitness  -> Mean: {np.mean(all_fitness):.5f} | Std: {np.std(all_fitness):.5f} | CV: {(np.std(all_fitness)/np.mean(all_fitness)):.3}",
            f"Features -> Mean: {np.mean(all_counts):.2f} | Std: {np.std(all_counts):.2f}"
        ])

        root = Path(__file__).resolve().parent.parent.parent
        
        path_results = os.path.join(root, "multilsbnew","resultados.csv")
        results = pd.read_csv(path_results, sep=",")
        lugNP = np.array(results["LugNP"])
        amh =  np.array(results["AMH"])
        hmpga = np.array(results["HMPGA"])
        heuristic = np.array(results["Heuristic"])
        instance = int(config["model_label"])

        if config["dataset_name"] == "SET2":
            instance = instance + 30
        elif config["dataset_name"] == "SET3":
            instance = instance + 60
        elif config["dataset_name"] == "SET4":
            instance = instance + 90

        dev_lugNP = (all_fitness[idx_min_fit] - lugNP[instance-1])/lugNP[instance-1] * 100
        dev_amh = (all_fitness[idx_min_fit] - amh[instance-1])/amh[instance-1] * 100
        dev_hmpga = (all_fitness[idx_min_fit] - hmpga[instance-1])/hmpga[instance-1] * 100
        dev_heuristic = (all_fitness[idx_min_fit] - heuristic[instance-1])/heuristic[instance-1] * 100

        finals_stats = self.saver.save_stats_fitness(stats["best_fits"])

        report.extend([
            "\n5. DEVs",
            "-" * 35,
            f"LugNP: {dev_lugNP}",
            f"AMH: {dev_amh}",
            f"HMPGA: {dev_hmpga}",
            f"Heuristic: {dev_heuristic}",
            "\n6. Stats",
            "-"*35,
            f"Invalid individuals: {np.mean(stats['infeasible'])} +/- {np.std(stats['infeasible'])}",
            f"Average Fitness Finals: {np.mean(stats['fitness_avg_final'])} +/- {np.std(stats['fitness_avg_final'])}",
            f"Average time to reach the minimum cost: {finals_stats['Time'].mean()} +/- {finals_stats['Time'].std()}",
            f"Average number of calls to the solver until the minimum cost is reached: {finals_stats['Calls'].mean()} +/-  {finals_stats['Calls'].std()}",
            f"Average number of generations until the minimum cost is reached: {finals_stats['Generation'].mean()} +/- {finals_stats['Generation'].std()}",

        ])

        file_path = self.saver.output_path / f"Final_Report_{config['dataset_name']}_{config['model_label']}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))

    
class PopulationAnalyzer:
    """
    Static tool to get the population fittest individual.
    """
    @staticmethod
    def get_fittest_individual(population):
        """Identifies the individual with the highest fitness value."""
        if not population:
            return None
        return min(population, key=lambda ind: ind.fitness if ind.fitness is not None and ind.fitness[1] else (1,float('inf')))