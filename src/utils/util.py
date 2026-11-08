"""
Utility Module for Data Loading and Preprocessing.

Handles the parsing of custom .dat files and manages the train/test splitting
logic for both classification and regression tasks.
"""

import numpy as np
from enum import Enum
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import os

class DatasetType(Enum):
    """Enumeration to distinguish between classification and regression problems."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Util:
    """
    Utility class providing static methods for data handling.
    """

    @staticmethod
    def load_dataset(root: str, test_set:str, instance: str) -> Optional[Tuple]:
        path = os.path.join(root, "multilsbnew", f"{test_set}")
        periods_path = os.path.join(path, "periods.txt")
        with open(periods_path, "r") as f:
            for linha in f:
                if linha.startswith("NT:"):
                    n_periods = int(linha.split(":")[1].strip()) #numero de periodos T
                    break
        #print(f"Períodos (T): {n_periods}")

        n_items = 78     #numero de itens J
        n_families = 17     #numero de familias F
        nbendp = 6  #numero de itens finais 

        end_itens = np.arange(0, nbendp) #Delta -> conjunto de itens finais
        families = np.arange(0, n_families)
        mc = np.array([1,2,8,9,10,17]) #ids das maquinas disponiveis

        demand_path = os.path.join(path, f"{instance}", "demand.txt")
        demand = np.loadtxt(demand_path, delimiter=";")
        avgdem = {}
        for i in range(nbendp):
            avgdem[i] = np.mean(demand.T[i])
        for i in range(nbendp, n_items):
            avgdem[i] = 0

        data_path = os.path.join(path, "multilsb_data.txt")
        with open(data_path, "r") as f:
            linhas = f.readlines()
            
            RU = float(linhas[0].split(":")[1].strip()) #fator de utilizacao de capacidade
            RUT = int(linhas[1].split(":")[1].strip()) #decide a formula de calculo de capacidae
            backlog_cost = float(linhas[2].split(":")[1].strip()) #penalidade de atraso
            
            
            a = {} #coeficiente de producao para o par item/maquina -> tempo ou recurso gasto por unidade
            for i in range(4, 82):
                partes = linhas[i].strip().split(";")
                item = int(partes[0]) - 1
                maquina = int(partes[1])
                coef = float(partes[2])
                a[(item, maquina)] = coef

            p = {} #mapeamento item -> familia
            for i in range(83, 119):
                partes = linhas[i].strip().split(";")
                item = int(partes[0]) - 1
                familia = int(partes[1]) - 1
                p[item] = familia

            st = {} #tempo de preparacao para o par familia/maquina
            for i in range(120, 131):
                partes = linhas[i].strip().split(";")
                familia = int(partes[0]) - 1
                maquina = int(partes[1])
                st[(familia, maquina)] = float(partes[2])   

            delta = {} #mapeamento de sucessor-> 1 se o item 2 eh sucessor de 1
            for i in range(132, 204):
                partes = linhas[i].strip().split(";")
                item1 = int(partes[0]) - 1
                item2 = int(partes[1]) - 1
                delta[(item1, item2)] = 1 

            e = {} #custos para cada produto
            for i in range(205, 283):
                partes = linhas[i].strip().split(";")
                item = int(partes[0]) - 1
                e[item] = float(partes[1])  


        r = {} #quantas unidades da peça i sao necessarias para fazer j
        for i in range(nbendp, n_items):
            for j in end_itens:
                r[(i, j)] = delta.get((i, j), 0.0)
                
            for j in end_itens:
                for k in range(nbendp, i): 
                    r[(i, j)] += delta.get((i, k), 0.0) * r.get((k, j), 0.0)

        #demanda media
        netdem = {}
        for i in range(nbendp):
            netdem[i] = avgdem[i]
        for i in range(nbendp, n_items):
            netdem[i] = avgdem[i] + sum(delta.get((i, j), 0.0) * netdem.get(j, 0.0) 
                                        for j in range(i))

        #capacidade dsponivel em cada maquina
        cap = {}
        for k in mc:
            if RUT == 1:
                cap[k] = (sum(a.get((i, k), 0.0) * netdem.get(i, 0.0) for i in range(n_items)) 
                        + sum(st.get((f, k), 0.0) for f in families) ) / RU
            else:
                cap[k] = (sum(a.get((i, k), 0.0) * netdem.get(i, 0.0) for i in range(n_items)) + 
                        (sum(st.get((f, k), 0.0) for f in families)) / RU)

        h = {} #custo de estoque agregado -> custo de fabrica + custo das pecas filhas
        h[n_items-1] = e[n_items-1]
        control = n_items
        while control > 0:
            control-=1
            h[control] = e[control] + sum(delta.get((j, control), 0.0) * h.get(j, 0.0) for j in range(control+1, n_items))

        totdem = {} #total da demanda por item -> big M
        big_m = 0
        for i in end_itens:
            totdem[i] = sum(demand[t, i] for t in range(n_periods))
            big_m += totdem[i]

        return {
            "n_periods": n_periods,
            "n_items": n_items,
            "n_families": n_families,
            "n_end_items": nbendp,
            "end_items": end_itens,
            "demand": demand,
            "hold_cost": h,
            "backlog_cost": backlog_cost,
            "item_family": p,
            "cap_machine": cap,
            "total_demands": totdem,
            "ids_machines": mc,
            "setup_time": st,
            "capacity_cost": a,
            "quantity_prod": r,
            "big_m": big_m,
            "direct_bom": delta,
        }