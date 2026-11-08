import argparse
import sys, gc, os
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline import GAPipeline
from src.utils.experiment_logger import ExperimentLogger
from src.utils.statistic import Statistics
from src.visualization.barplot import BarPlot
from src.visualization.graph import Graph
from src.visualization.values import SaveValues
from src.utils.util import Util
from src.core.gawll import GAwLL
from src.core.hmpga import Hmpga

def main():
    parser = argparse.ArgumentParser(description="GAwLL Framework: XAI-Enhanced Feature Selection")

    # --- GA definition ---
    parser.add_argument("--ga", default="GAwLL", help="Class of GA (default is GAwLL)")

    # --- Set Definition ---
    parser.add_argument("--sets", nargs='+', required=True,
                        help="Sets to process (e.g., SET1, SET2, SET3, SET4)")
    parser.add_argument("--instances", type=int, nargs='+', required=True,
                        help="Number of instances")
    
    # --- General Hyperparameters ---
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed for reproducibility")
    parser.add_argument("--n_jobs", type=int, default=6, help="Number of jobs")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of independent GA executions per experiment")
    parser.add_argument("--pop_size", type=int, default=100, help="Population size")
    parser.add_argument("--max_time", type=int, default=100.0, help="Maximum time of training")
    parser.add_argument("--cross_rate", type=float, default=0.8, help="Crossover probability")

    # --- GAwLL Hyperparameters ---
    parser.add_argument("--max_gen", type=int, default=20,
                        help="Maximum number of generations")
    parser.add_argument("--tau_reset", type=int, default=50,
                        help="Generations without improvement before population reset")
    parser.add_argument("--freq_fix", type=int, default=5,
                            help="Frequency for fix-and-optimize (GAwLL)")
    parser.add_argument("--elite_rate", type=float, default=0.05,
                            help="Elite rate for fix-and-optimize (GAwLL)")
    parser.add_argument("--random_rate", type=float, default=0.01,
                            help="Random rate for fix-and-optimize (GAwLL)")

    # --- HMPGA Hyperparameters ---
    parser.add_argument("--n_trees", type=int, default=3, help="Number of trees (HMPGA)")
    parser.add_argument("--mut_rate", type=float, default=0.95, help="Mutation probability (HMPGA)")
    

    # --- XAI and Linkage Settings (Opt-out Logic) ---
    parser.add_argument("--no_linkage", action="store_false", dest="linkage_learning",
                        help="Deactivate Linkage Learning (default is True)")
    parser.set_defaults(linkage_learning=True)

    # --- Output ---
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of top features to display in reports")
    
    args = parser.parse_args()
    raiz = Path(__file__).resolve().parent.parent
    if args.ga.lower() == "gawll":
        model = GAwLL
    elif args.ga.lower() == "hmpga":
        model = Hmpga
    else:
        print("[ERRO] Invalid model! Valid models: gawll ou hmpga.")
        return
    # tests_sets = ["SET1"]
    # instances = ["01"]
    instances = []

    if len(args.instances) == 1:
        i = args.instances[0]
        if i/10 < 1:
            instances = ['0'+str(i)]
        else:
            instances = [str(i)]
    elif len(args.instances) == 2:
        for i in range(args.instances[0],args.instances[1]):
            name = ""
            if i/10 < 1:
                name = '0'+str(i)
            else:
                name = str(i)
            instances.append(name)

    for test_set in args.sets:
        for instance in instances:
            params = Util.load_dataset(raiz, test_set, instance)
            
            vars_names = []
            for i in range(params["n_periods"]):
                for j in range(params["n_families"]):
                    vars_names.append(f"P{i}:F{j}")

            if args.ga.lower() == "gawll":
                mut_rate = 1.0 / (params["n_periods"] * params["n_families"])
            else:
                mut_rate = args.mut_rate
    
            db_logger = ExperimentLogger()
            config = {
                'seed': args.seed,
                'n_runs': args.n_runs,
                'variables': vars_names,
                'model_label': instance,
                'dataset_name': test_set,
                'n_jobs': args.n_jobs,
                'top_n': params["n_periods"] * params["n_families"],
                'compare_methods': False,
                'ga_params': {
                    'pop_size': args.pop_size,
                    'max_generations': args.max_gen,
                    'crossover_rate': args.cross_rate,
                    'tau_reset': args.tau_reset,
                    'linkage_learning': args.linkage_learning,
                    'max_time': args.max_time,
                    'n_trees': args.n_trees,
                    'mutation_probability': mut_rate,
                    'freq_fix': args.freq_fix,
                    'elite_rate': args.elite_rate,
                    'random_rate': args.random_rate,
                }
            }
    
            print(f"\n" + "═" * 75)
            print(f" STARTING EXPERIMENT {test_set}: {instance}")
            print(f" CONFIG: {args.ga.upper()} - Linkage={args.linkage_learning}")
            print("═" * 75)
    
            if args.linkage_learning:
                if args.ga.lower() == "gawll":
                    path_results = f'ga/{test_set}/results-experiment-{test_set}-{instance}/'
                else:
                    path_results = f'{args.ga.lower()}-ll/results-experiment-{test_set}-{instance}/'
            else:
                path_results = f'teste-{args.ga.lower()}/{test_set}/results-experiment-{test_set}-{instance}/'

            stats = Statistics(BarPlot(output_dir=path_results), 
                                            Graph(output_dir=path_results), 
                                            SaveValues(output_dir=path_results), 
                                            db_logger)
            pipeline = GAPipeline(stats, model)
            
            try:
                pipeline.run_experiment(params, config)
            except Exception as e:
                import traceback
                print(f"\n    [CRITICAL ERROR] Experiment failed: {e}")
                traceback.print_exc()
            finally:
                # Force memory cleanup between different model/dataset pairs
                del pipeline
                del stats
                gc.collect()
    
    print("\n" + "═" * 75)
    print(" ALL SCHEDULED EXPERIMENTS COMPLETED SUCCESSFULLY.")
    print("═" * 75)
            
            
if __name__ == "__main__":
    main()

