---

## 📌 Sobre o Projeto

Este repositório contém os algoritmos, protocolos experimentais e scripts de análise estatística desenvolvidos para a resolução do **Multi-Level Capacitated Lot-Sizing Problem (MLCLSP) com Backlogging** utilizando o benchmark **MULTI-LSB**.

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
  - `scipy`, `scikit-posthocs` — Testes de hipótese (Wilcoxon, Shapiro-Wilk, Friedman)
  - `matplotlib`, `seaborn` — Geração de heatmaps e plots estatísticos
  - Solver de PLI (ex: CPLEX / Gurobi / CBC)

---

## 🚀 Como Executar

### 1. Clonar o Repositório
```bash
git clone [https://github.com/seu-usuario/](https://github.com/seu-usuario/){repo_name}.git
cd {repo_name}