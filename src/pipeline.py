import time
import multiprocessing
import gc
import traceback
import psutil
import numpy as np
from typing import List, Dict
from src.core.fitness_functions import solve_model_parallel, worker_initalize


def worker_perf_task(task_data: tuple) -> float:
    setup = task_data
    try:
        # start = time.time()
        result, _ = solve_model_parallel(setup)
        # elapsed = time.time() - start
        # print(
        #     f"[WORKER] PID={multiprocessing.current_process().pid} "
        #     f"tempo={elapsed:.2f}s",
        #     flush=True
        # )
        return result
    except Exception as e:
        traceback.print_exc()
        return 0.0

class GAPipeline:
    def __init__(self, stats_manager, ga):
            """
            Initializes the pipeline with a statistics manager.
            """
            self.stats = stats_manager
            self.db_logger = stats_manager.db_logger
            self.ga = ga
    def run_experiment(self, params: Dict, config: Dict):
        # Accumulators for multi-run averages
        sum_vint, sum_vimp = None, None
        count_success = 0
        best_ind_run = []

        id_group = self.db_logger.register_experiment(
            "", "", config['n_runs']
        )

        is_linkage_enabled = config['ga_params']['linkage_learning']
        ctx = multiprocessing.get_context('spawn')
        pool = ctx.Pool(processes=config['n_jobs'], maxtasksperchild=100, initializer=worker_initalize, initargs=(params,))
        stats_avg = {
            "fitness_avg_final": [],
            "exec_time": [],
            "infeasible": [],
            "best_fits": [],
        }
        # try:
        for r in range(config['n_runs']):
            current_seed = config['seed'] + r
            print(f"\n >>> Run {r + 1}/{config['n_runs']} (Seed: {current_seed})\n")
            
            def batch_evaluator(feature_masks: List[np.ndarray]) -> List[float]:
                """
                Parallel evaluator with dynamic RAM protection (Safety Lock).
                """
                
                try:
                    mem_percent = psutil.virtual_memory().percent
                    if mem_percent > 90:
                        print(
                            f"\n[RAM] Uso elevado antes do batch: "
                            f"{mem_percent:.1f}%"
                        )
                    chunk_size = max(1,len(feature_masks) // (config['n_jobs'] * 2))

                    result_async = pool.map_async(worker_perf_task, feature_masks,chunksize=chunk_size)

                    return result_async.get(timeout=3600)

                except Exception as e:
                    print(f"[ERROR] Mapping failed: {e}")
                    traceback.print_exc()
                    return [0.0] * len(feature_masks)

            # --- EXECUTION BLOCK ---
            # try:
            chrm_size = params["n_periods"] * params.get("n_families", params["n_items"]) 
            # 2. GA Execution
            ga = self.ga(perf_batch=batch_evaluator,
                        chrom_size=chrm_size,
                        params = params,
                        **config['ga_params'])

            start_ga = time.time()
            ga.run(current_seed)
            t_ga = time.time() - start_ga
            print(f"\nTime: {t_ga}")
            stats_avg["exec_time"].append(t_ga)

            res_fs = ga.get_result()
            best_ind_run.append(res_fs)
            print(f"Best setup: {res_fs[0]}, Fitness: {res_fs[1]:.2f}, Feasible: {res_fs[2]}")

            stats_avg["infeasible"].append(ga.stats["infeasible"])
            stats_avg["best_fits"].append(ga.stats["best_fits"])
            stats_avg["fitness_avg_final"].append(ga.stats["fitness_avg_final"])
            
            # Results from the Linkage Learning Mutation (VInt and VImp)
            if is_linkage_enabled:
                vimp_vector = ga.importance.export_importance_vector()
                vint_matrix = ga.evig.export_interaction_matrix()

                # Sum the VImp and VInt to obtain the average of VInt and VImp from different methods
                if sum_vimp is None:
                    sum_vimp, sum_vint = vimp_vector, vint_matrix
                else:
                    np.add(sum_vimp, vimp_vector, out=sum_vimp)
                    np.add(sum_vint, vint_matrix, out=sum_vint)
                
                count_success += 1
            try:
                peak_ram_usage = psutil.Process().memory_info().rss / (1024 * 1024)
            except:
                peak_ram_usage = 0.0
            
                
            # SQL Logging
            run_data = {
                'run': r + 1, 'time_gawll': t_ga,
                'time_perm': 0, 'time_intr': 0,
                'peak_ram': peak_ram_usage
            }
            self.db_logger.log_run(id_group, run_data)

            ga.cleanup()
            del ga          
            for _ in range(3):
                gc.collect()
            time.sleep(3)
                
                
            # except Exception as e:
            #     print(f"    [ERROR Run {r + 1}]: {e}")
                
            # finally:
            #     gc.collect()
            #     time.sleep(3) 
        # except KeyboardInterrupt:
        #     print("\n[STOP] Experimento interrompido pelo usuário.")
        #     pool.terminate()
        #     pool.join()
        #     raise
        # except Exception:
        #     pool.terminate()
        #     pool.join()
        #     raise
        # else:
        #     pool.close()
        #     pool.join()

        for k in stats_avg.keys():
            stats_avg[k] = np.array(stats_avg[k])
        
        if not is_linkage_enabled:
            self.stats.save_feature_selection_statistics(best_ind_run, config, stats_avg)
            self.stats.db_logger.finalize_summary(id_group)
        
        elif is_linkage_enabled and count_success > 0:
            aggregates = {
                'id_group': id_group,
                'vint_gawll': sum_vint / count_success,
                'vimp_gawll': sum_vimp / count_success,
                'vimp_perm': None,
                'vimp_intr': None
            }
            self.stats.save_all_results(best_ind_run, aggregates, config, stats_avg) 
    