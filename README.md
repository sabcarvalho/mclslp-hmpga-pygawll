# Repositório do Trabalho de Conclusão de Curso: Algoritmo Genético com Linkage Learning Aplicado ao Problema de Dimensionamento de Lotes Multinível Capacitado com Atrasos

---

## 📌 Sobre o Projeto

Este repositório contém os algoritmos, protocolos experimentais e scripts de análise estatística desenvolvidos para a resolução do **Multi-Level Capacitated Lot-Sizing Problem (MLCLSP) com Backlogging** utilizando o benchmark **MULTI-LSB**. Adaptação do PyGAwLL (https://github.com/dgabrl/PyGAwLLfs).

### Algoritmos Implementados:
- **HMPGA (Reimplementado):** Algoritmo Genético Híbrido Multinível em Python.
- **HMPGA-LL:** HMPGA estendido com aprendizado de ligação entre variáveis (*Linkage Learning*).
- **GAwLL + Fix & Optimize:** Algoritmo com decomposição exata/meta-heurística via solver de PLI.

---

## 🛠️ Tecnologias e Dependências

- **Linguagem:** Python 3.10+
- **Framework Base:** `pygawll`
- **Bibliotecas Principais:**
  - `pandas`, `numpy` — Processamento de dados e estatísticas descritivas
  - `matplotlib`, `seaborn` — Geração de heatmaps e plots estatísticos
  - `gurobipy` — Solver de Programação Linear

---
## 🏗 Estrutura do Repositório 

### 📂 `src/core/`
* **Classes principais dos Algoritmos Genéticos (GAwLL e HMPGA):** lógica de geração de populações e controle de gerações.
* **Operadores dos algoritmos genéticos:** lógica de seleção, mutação e crossover.
* **Mutação com Linkage Learning:** O mecanismo que constrói grafos de interação e estima $VInt$ e $VImp$ durante a fase de mutação.
* **Funções de Fitness**: Definição dos modelos, restrições e lógica da heurística Fix-and-Optimize.
* **Pipeline:** Orquestra execuções independentes, gerencia recursos de hardware e implementa o **RAM Escalator**

### 📂 `src/visualization/` & `src/utils/`
* **Visualization:** Gera grafos de interação completos/reduzidos (NetworkX), gráficos de barras (brutos/normalizados) e métricas em arquivos `.txt` e `.csv`.
* **Utils:** Carregamento de dados.

### 📂 `multilsbnew`
* **Conjuntos e Instâncias de teste do **MULTI-LSB**. Fonte: https://pureportal.strath.ac.uk/en/datasets/multilsb-multi-item-lot-sizing-with-backlogging/

### 📂 `gawll`, `hmpga` & `hmpga-ll`
* **Resultados das execuções:** Para cada algoritmo, separado por conjunto, os resultados apresentados e analisados na monografia.

---

## 🚀 Como Executar

### 1. Clonar o Repositório
```bash
git clone https://github.com/sabcarvalho/mclslp-hmpga-pygawll.git
cd mclslp-hmpga-pygawll
```
### 2. Instalar Dependências
```bash
pip install -r requirements.txt
```
### 3. Executar 
```bash
python src\multilsb.py --ga hmpga --set SET1 --instances 1 30 --pop_size 13 --n_runs 10 --max_time 100 --cross_rate 1.0
```
### 4. Parâmetros
**Parâmetros obrigatórios**
* `--sets`: Conjuntos de instâncias do benchmark a processar (ex: SET1 SET2).
* `--instances`: Identificador ou número de instâncias a processar (números entre 1 e 31):
  * Um número informado: testes apenas em uma instância. Exemplo: `--instances 1`
  * Dois números informados: testes em todas as instâncias na sequência determinada pelos dois números. Exemplo `--instances 1 30`

**Parâmetros opcionais**
| Argumento | Padrão | Descrição |
| :--- | :----: | :--- |
| `--ga` | gawll | Classe/variante do Algoritmo Genético a ser executado. |
| `--seed` | 42 | Semente aleatória base para reprodutibilidade dos experimentos. |
| `--n_jobs` | 6 | Número de processos/threads paralelas para execução. |
| `--n_runs` | 1 | Número de execuções independentes do GA por experimento. |
| `--pop_size` | 100 | Tamanho da população de indivíduos. |
| `--max_time` | 100 | Tempo máximo de treinamento/execução (em segundos). |
| `--cross_rate` | 0.8 | Probabilidade de ocorrência do operador de cruzamento (crossover). |
| `--no_linkage` | False | Desativa o Linkage Learning (por padrão, o aprendizado de ligação é True). |
| `--top_n` | 10 | Número de variáveis/características mais importantes exibidas nos relatórios. |
| :--- | GAwLL | :--- |
| `--tau_reset` | 50 | Gerações sem melhoria na aptidão antes de reiniciar a população. |
| `--freq_fix` | 5 | Frequência (em gerações) para aplicação da heurística Fix-and-Optimize. |
| `--elite_rate` | 0.05 | Taxa de indivíduos de elite selecionados para o Fix-and-Optimize. |
| `--random_rate` | 5 | Taxa de indivíduos aleatórios para o Fix-and-Optimize. |
| :--- | HMPGA | :--- |
| `--n_trees` | 3 | Número de árvores na estrutura multinível do algoritmo |
| `--mut_rate` | 0.95 | Probabilidade de ocorrência do operador de mutação |

