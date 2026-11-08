import gurobipy as gp
import numpy as np
import time

def worker_initalize(params):
    global worker_model, worker_vars, params_g
    worker_vars = {}
    params_g = params
    
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    
    worker_model = gp.Model("MULTI-LOTSB-WORKER", env=env)
    worker_model.Params.Threads = 1 
    
    n_per = params["n_periods"]
    n_fam = params["n_families"]
    
    fam_setup = worker_model.addVars(n_per, n_fam, vtype=gp.GRB.BINARY, name="fam_setup")
    production = worker_model.addVars(n_per, params["n_items"], vtype=gp.GRB.CONTINUOUS)
    stock = worker_model.addVars(n_per, params["n_items"], vtype=gp.GRB.CONTINUOUS)
    backlog = worker_model.addVars(n_per, params["n_end_items"], vtype=gp.GRB.CONTINUOUS)

    worker_vars['fam_setup'] = fam_setup
    worker_vars["production"] = production

    worker_model.setObjective(gp.quicksum(params["hold_cost"].get(j, 0) * stock[t, j] 
                                        for t in range(params["n_periods"]) for j in range(params["n_items"])) +
                            gp.quicksum(params["backlog_cost"] * params["hold_cost"].get(j, 0) * backlog[t, j] 
                                        for t in range(params["n_periods"]) 
                                        for j in params["end_items"]), sense=gp.GRB.MINIMIZE)
    
    # restricoes
    #restricao de inventorio -> itens finais (2)
    worker_model.addConstrs((production[0, j] + backlog[0, j] == stock[0, j] + params["demand"][0, j] 
                    for j in params["end_items"]))
    worker_model.addConstrs((stock[t-1, j] + backlog[t, j] + production[t, j] == stock[t, j] + 
                        backlog[t-1, j] + params["demand"][t, j] 
                    for t in range(1, params["n_periods"]) for j in params["end_items"]))

    #restricao de inventorio -> subcomponentes (3)
    worker_model.addConstrs((production[0, j] == stock[0, j] + 
                    gp.quicksum(params["direct_bom"].get((j, k), 0.0) * production[0, k]
                                for k in range(params["n_items"]))
                    for j in range(params["n_end_items"], params["n_items"])))
    worker_model.addConstrs((stock[t-1, j] + production[t, j] == stock[t, j] + 
                    gp.quicksum(params["direct_bom"].get((j, k), 0.0) * production[t, k] 
                                for k in range(params["n_items"])) 
                    for t in range(1, params["n_periods"]) for j in range(params["n_end_items"], params["n_items"])))

    #restricoes limite superior (11)
    #limite guiado pela demanda maxima -> itens finais
    for i in range(params["n_end_items"]):
        f = params["item_family"].get(i)
        for t in range(params["n_periods"]):
            setup_limit = fam_setup[t, f] if f is not None else 1
            backlog_prev = backlog[t-1, i] if t >= 1 else 0
            
            worker_model.addConstr(production[t, i] <= params["total_demands"][i] * setup_limit + backlog_prev)
    for i in range(params["n_end_items"]):
        f = params["item_family"].get(i)
        for t in range(n_per):
            # fam_setup[t, f] é a variável matemática, não um número
            setup_limit = fam_setup[t, f] if f is not None else 1 
            backlog_prev = backlog[t-1, i] if t >= 1 else 0
            worker_model.addConstr(production[t, i] <= params["total_demands"][i] * setup_limit + backlog_prev)
            
    #limite guiado pela demanda maxima dependente -> subcomponentes
    for i in range(params["n_end_items"], params["n_items"]):
        f = params["item_family"].get(i)
        for t in range(params["n_periods"]):
            setup_limit = fam_setup[t, f] if f is not None else 1
            
            demand_sum = gp.quicksum(params["quantity_prod"].get((i, j), 0.0) * (params["total_demands"][j] * setup_limit + 
                                                            (backlog[t-1, j] if t >= 1 else 0)) 
                                    for j in range(params["n_end_items"]))
            worker_model.addConstr(production[t, i] <= demand_sum)

    #limite guiado pela capacidade residual da maquina
    for i in range(params["n_items"]):
        f = params["item_family"].get(i)
        for k in params["ids_machines"]:
            if params["capacity_cost"].get((i, k), 0.0) > 0:
                for t in range(params["n_periods"]):
                    if f is not None:
                        limite_cap = (params["cap_machine"][k] - params["setup_time"].get((f, k), 0.0)) * fam_setup[t, f]
                    else:
                        limite_cap = params["cap_machine"][k]
                    worker_model.addConstr(params["capacity_cost"].get((i, k), 0.0) * production[t, i] <= limite_cap)

    #restricao capacidade maquina (12)
    worker_model.addConstrs((gp.quicksum(params["capacity_cost"].get((j, k), 0.0) * production[t, j] for j in range(params["n_items"])) <= 
                    params["cap_machine"][k] - gp.quicksum(params["setup_time"].get((f, k), 0.0) * fam_setup[t, f] for f in range(params["n_families"])) 
                                for k in params["ids_machines"] for t in range(params["n_periods"])))
    
    # Atualiza o modelo para compilar a estrutura interna
    worker_model.update()

def solve_model_parallel(setup, fix = None, return_lot_sizes = False):
    global worker_model, worker_vars, params_g
    n_per = params_g["n_periods"]
    n_fam = params_g["n_families"]
    fam_setup = worker_vars['fam_setup']
    for i in range(n_per * n_fam):
        t = i // n_fam
        f = i % n_fam
        if fix is None:
            fam_setup[t, f].lb = setup[i]
            fam_setup[t, f].ub = setup[i]
        else:
            if i in fix:
                fam_setup[t, f].lb = setup[i]
                fam_setup[t, f].ub = setup[i]
            else:
                fam_setup[t, f].lb = 0
                fam_setup[t, f].ub = 1
    worker_model.optimize()
    sol_setup = setup.copy() if isinstance(setup, np.ndarray) else np.array(setup)
    if worker_model.status == gp.GRB.OPTIMAL or (worker_model.status == gp.GRB.TIME_LIMIT and worker_model.SolCount > 0):
        if fix is not None:
            sol_setup = np.array([[round(fam_setup[i, j].X) for j in range(n_fam)] for i in range(n_per)]).reshape(n_per * n_fam)

        lot_sizes = None
        if return_lot_sizes:
            x_vars = worker_vars["production"]
            lot_sizes = np.zeros((n_per, params_g["n_items"]))
            for t in range(n_per):
                for j in range(params_g["n_items"]):
                    lot_sizes[t, j] = x_vars[t, j].X
                    
        result = {"fitness": worker_model.objVal, "feasible": True, "violation": 0.0}
        
        return (result, sol_setup, lot_sizes) if return_lot_sizes else (result, sol_setup)
    if worker_model.status == gp.GRB.INFEASIBLE: 
        worker_model.feasRelaxS(0, False, False, True) 
        worker_model.optimize() 
        if worker_model.status == gp.GRB.OPTIMAL: 
            violacao = worker_model.objVal 
            return {"fitness": None, "feasible": False, "violation": violacao * params_g.get("big_m", 1000)}, sol_setup     
    return {"fitness": None, "feasible": False, "violation": params_g.get("big_m", 1000) * 1000}, sol_setup

def fix_and_optimize(individual, period_window_size = 2, family_window_size = 2, init_train=100, max_time=100):
    global params_g
   
    len_chrom = params_g["n_periods"]*params_g["n_families"]

    #matriz de indices: indexes[i][j] -> posicao da variavel do periodo i da familia j no vetor do cromossomo
    indexes = np.arange(len_chrom).reshape(params_g["n_periods"], params_g["n_families"])
    fixed = set(indexes.flatten().tolist())#todas as variaveis comecam fixas
    for t in range(params_g["n_periods"]): #periodo t: todas as familias nesse periodos livres
        fixed = fixed - set(indexes[t].tolist())
        if t >= period_window_size:
            #o periodo que fica para tras quando a janela sai do comeco eh fixo tbm
            fixed = fixed | set(indexes[t-period_window_size].tolist())
        result, sol = solve_model_parallel(individual.chromosome, fix=fixed)
        individual.chromosome = sol
        if ((time.perf_counter() - init_train) >= max_time): 
            if result["feasible"]:
                individual.fitness = (0, result["fitness"])
            else:
                individual.fitness = (1, result["violation"])
            return

    #matriz de indices transposta: indexes[j][i] -> posicao da variavel do periodo i da familia j no vetor do cromossomo
    indexes = indexes.T
    fixed = set(indexes.flatten().tolist())
    for f in range(params_g["n_families"]):
        fixed = fixed - set(indexes[f].tolist())
        if f >= family_window_size:
            fixed = fixed | set(indexes[f-family_window_size].tolist())
        result, sol = solve_model_parallel(individual.chromosome, fix=fixed)
        individual.chromosome = sol
        if ((time.perf_counter() - init_train) >= max_time): 
            if result["feasible"]:
                individual.fitness = (0, result["fitness"])
            else:
                individual.fitness = (1, result["violation"])
            return

    if result["feasible"]:
        individual.fitness = (0, result["fitness"])
    else:
        individual.fitness = (1, result["violation"])

def fix_and_optimize_oriented(individual, imp_fam, imp_per, int_fam, int_per, 
                              period_window_size = 2, family_window_size = 2, top=0.7):
    global params_g
    top_fam = int(imp_fam.shape[0] * top)
    top_per = int(imp_per.shape[0] * top)
    len_chrom = params_g["n_periods"]*params_g["n_families"]
    
    indexes = np.arange(len_chrom).reshape(params_g["n_periods"], params_g["n_families"])
    fixed = set(indexes.flatten().tolist()) 

    idx_ordenated = np.argsort(imp_per)[::-1]
    for i in range(top_per):
        top_int = np.argsort(int_per[idx_ordenated[i]])[::-1][:period_window_size-1]
        fixed = fixed - set(indexes[idx_ordenated[i]].tolist())
        for t in top_int:
            fixed = fixed - set(indexes[t].tolist())
        _, sol = solve_model_parallel(individual.chromosome, fix=fixed)
        individual.chromosome = sol
        fixed = fixed | set(indexes[idx_ordenated[i]].tolist())
        for t in top_int:
            fixed = fixed | set(indexes[t].tolist())


    indexes = indexes.T
    fixed = set(indexes.flatten().tolist()) 
    idx_ordenated = np.argsort(imp_fam)[::-1]
    for i in range(top_fam):
        top_int = np.argsort(int_fam[idx_ordenated[i]])[::-1][:family_window_size-1]
        fixed = fixed - set(indexes[idx_ordenated[i]].tolist())
        for t in top_int:
            fixed = fixed - set(indexes[t].tolist())
        result, sol = solve_model_parallel(individual.chromosome, fix=fixed)
        individual.chromosome = sol
        fixed = fixed | set(indexes[idx_ordenated[i]].tolist())
        for t in top_int:
            fixed = fixed | set(indexes[t].tolist())

    if result["feasible"]:
        individual.fitness = (0, result["fitness"])
    else:
        individual.fitness = (1, result["violation"])