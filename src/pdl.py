import argparse
import sys
import gc
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline import GAwLLPipeline
from src.utils.experiment_logger import ExperimentLogger
from src.utils.statistic import Statistics
from src.visualization.barplot import BarPlot
from src.visualization.graph import Graph
from src.visualization.values import SaveValues

def main():
    parser = argparse.ArgumentParser(description="GAwLL Framework: XAI-Enhanced Feature Selection")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed for reproducibility")

    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of independent GA executions per experiment")
    parser.add_argument("--pop_size", type=int, default=100,
                        help="Population size")
    parser.add_argument("--max_gen", type=int, default=200,
                        help="Maximum number of generations")
    parser.add_argument("--cross_rate", type=float, default=0.8,
                        help="Crossover probability")
    parser.add_argument("--tau_reset", type=int, default=50,
                        help="Generations without improvement before population reset")

    # --- XAI and Linkage Settings (Opt-out Logic) ---
    parser.add_argument("--no_linkage", action="store_false", dest="linkage_learning",
                        help="Deactivate Linkage Learning (default is True)")
    parser.set_defaults(linkage_learning=True)

    # --- Output and Hyperparameters ---
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of top features to display in reports")
    
    args = parser.parse_args()

    

    # setup esperado: [1,1,0,1]
    # custo minimo: 542
    p1 = {
            "n_items": 1,
            "n_periods": 4,
            "big_m": 1000,
            "custo_setup":  [150],
            "custo_estoque": [2],
            "demandas": [[104],[174],[46],[112]],
            "custo_otimo": 542,
        }

    # setup esperado: [ 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    # custo minimo: 7885
    p2 = {
            "custo_otimo": 7885,
            "n_items": 2,
            "n_periods": 12,
            "big_m": 3210,
            "estoque_inicial":[300, 120],
            "estoque_final":[300, 200],
            "recursos_produto": [0.2 , 0.4],
            "custo_setup":  [500, 300],
            "custo_estoque": [2,3],
            "recursos_disp": [100, 160, 120, 80, 80, 80, 120, 140, 160, 160, 160, 160],
            "demandas": [[185, 210], [250, 0], [0, 165], [120, 0], [175, 330], [210, 0],
                         [0, 0], [0, 240], [180, 55], [260, 70], [0, 120], [140, 0]
                        ]
        }
    
    # setup esperado: [1, 0, 1, 0, 1, 1, 1, 1]
    # custo minimo: 36000
    p3 = {
        "custo_otimo": 36000,
        "n_items": 1,
        "n_periods": 8,
        "big_m": 15555,
        "custo_setup": [5000],
        "custo_estoque": [5],
        "demandas": [[400],[400],[800],[800],[1200],[1200],[1200],[1200]]
    }

    # setup esperado:Best setup: [1,1,0,0,1,1,1,0,0,0,0,1,0,0,0,1] 
    # custo minimo: 1323300
    p4 = {
        "custo_otimo": 1323300,
        "n_items": 2,
        "n_periods": 8,
        "big_m": 17200,
        "custo_setup": [5000, 6000],
        "custo_prod": [[100,200], [100,200], [100,200], [100,200], [150,200], [150,200], [150,200], [150,200]],
        "custo_estoque": [5, 6],
        "recursos_setup": [0.5],
        "recursos_disp": [10, 10, 10, 10, 10, 10, 10, 10],
        "demandas": [[400, 200],[400, 300],[800,800],[800,200],[500,400],[700,800],[400,500],[800,600]]
    }
    experimentos = [p1, p2, p3, p4]
    for exp in range(len(experimentos)):
        vars_names = []
        for i in range(experimentos[exp]["n_periods"]):
            for j in range(experimentos[exp]["n_items"]):
                vars_names.append(f"Pe{i}:Prod{j}")

        db_logger = ExperimentLogger()
        config = {
            'seed': args.seed,
            'n_runs': args.n_runs,
            'variables': vars_names,
            'model_label': "",
            'dataset_name': "",
            'top_n': 4,
            'compare_methods': False,
            'ga_params': {
                'pop_size': args.pop_size,
                'max_generations': args.max_gen,
                'crossover_rate': args.cross_rate,
                'tau_reset': args.tau_reset,
                'linkage_learning': args.linkage_learning
            }
        }

        print(f"\n" + "═" * 75)
        print(f" STARTING EXPERIMENT {exp}")
        print(f" CONFIG: Linkage={args.linkage_learning}")
        print("═" * 75)

        if args.linkage_learning:
            path_results = f'gawll-results-experiment-{exp}/'
        else:
            path_results = f'ga-results-experiment-{exp}/'
        stats = Statistics(BarPlot(output_dir=path_results), 
                                        Graph(output_dir=path_results), 
                                        SaveValues(output_dir=path_results), 
                                        db_logger)
        pipeline = GAwLLPipeline(stats)
        try:
            pipeline.run_experiment(experimentos[exp], config)
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