# Grid Search Module - BERT-PLI

Módulo Python profissional para execução de busca em grade de hiperparâmetros.

## 🚀 Instalação

O módulo já está integrado ao projeto. Nenhuma instalação adicional é necessária.

## 📖 Uso Rápido

### Modo CLI

```powershell
# Teste rápido (8 experimentos)
python -m gridsearch.core --config config/experiments/BertPLI.config \
                          --search-config gridsearch/config/grid_search_test.json \
                          --parallel 2

# Execução completa (216 experimentos)
python -m gridsearch.core --config config/experiments/BertPLI.config \
                          --search-config gridsearch/config/grid_search.json \
                          --parallel 2

# Retomar execução interrompida
python -m gridsearch.core --resume --parallel 2

# Análise de resultados
python -m gridsearch.core --analyze-only
```

### Modo PowerShell

```powershell
# Teste
.\gridsearch\scripts\run_grid_search.ps1 -Mode test -Parallel 2

# Completo
.\gridsearch\scripts\run_grid_search.ps1 -Mode full -Parallel 2

# Retomar
.\gridsearch\scripts\run_grid_search.ps1 -Mode resume -Parallel 2

# Análise
.\gridsearch\scripts\run_grid_search.ps1 -Mode analyze
```

### Modo Python (importável)

```python
from gridsearch import run_grid_search
import json

# Carrega configuração
with open('gridsearch/config/grid_search_test.json') as f:
    grid_config = json.load(f)

# Executa
results = run_grid_search(
    base_config_path='config/experiments/BertPLI.config',
    grid_config=grid_config['hyperparameters'],
    parallel=2
)

# Analisa
from gridsearch import analyze_results
analysis = analyze_results(results)
```

## 📁 Estrutura

```sh
gridsearch/
├── __init__.py              # Exports do módulo (v1.1.0)
├── core.py                  # Motor de execução
├── utils.py                 # Validações de memória
├── analysis.py              # Análise escalar de resultados
├── skyband.py               # Motor de Skyband Query (multi-critério)
├── config/
│   ├── grid_search.json          # Grid completo (216 exp)
│   ├── grid_search_test.json     # Grid de teste (8 exp)
│   ├── grid_search_minimal.json  # Grid minimal (3 exp)
│   ├── grid_search_multienv.json # Grid multi-ambiente (1080 exp)
│   └── sla_profiles.json         # Perfis de SLA (6 perfis × 5 ambientes)
└── docs/
    ├── GRIDSEARCH.md        # Este documento
    ├── OVERVIEW.md          # Visão geral
    └── PIPELINE.md          # Pipeline completo
```

## ⚙️ Configurações

### Grid de Teste (8 experimentos)

- Learning rates: [1e-5, 2e-5]
- Batch sizes: [8, 16]
- Otimizadores: ["adam"]
- **Tempo:** ~2-3 horas com parallel=2
- **Memória:** ~7-10 GB

### Grid Completo (216 experimentos)

- Learning rates: [1e-5, 2e-5, 3e-5, 5e-5]
- Batch sizes: [8, 16, 32]
- Otimizadores: ["adam", "adamw"]
- Dropouts: [0.1, 0.2, 0.3]
- Seeds: [42, 123, 456]
- **Tempo:** ~72-108 horas com parallel=2
- **Memória:** 32GB RAM recomendado

## 🔍 Análise de Resultados

Os resultados são salvos automaticamente em:

- `output/experiments/grid_search/grid_search_results.json` - Resultados completos
- `output/experiments/grid_search/grid_search_summary.txt` - Resumo legível
- `output/experiments/grid_search/analysis/` - Análises detalhadas

### Critérios de Análise

O módulo identifica as melhores configurações por **5 critérios diferentes**:

1. **⏱️ Tempo de Treinamento** (`train_time_sec`)
   - Menor tempo é melhor
   - Métrica: segundos

2. **⚡ Eficiência Energética** (`energy_kwh`)
   - Menor consumo é melhor
   - Métrica: kWh (quilowatt-hora)

3. **⚡ Throughput Computacional** (`total_gflops`)
   - Menor overhead é melhor (modelos mais leves)
   - Métrica: GFLOPS (bilhões de operações de ponto flutuante por epoch)

4. **🌍 Emissão de Carbono** (`emissions_kg_co2`)
   - Menor emissão é melhor
   - Métrica: kg CO₂
   - Calculado via CodeCarbon

5. **💰 Custo Financeiro** (`cost_usd`)
   - Menor custo é melhor
   - Métrica: USD (dólares americanos)
   - Calculado: `energy_kwh × tarifa_energia`
   - Tarifa padrão: $0.12/kWh (configurável)

### Configurando a Tarifa de Energia

```bash
# Linux / macOS
export ENERGY_COST_USD_PER_KWH=0.15
python -m gridsearch.core --config ... --parallel 2

# Windows PowerShell
$env:ENERGY_COST_USD_PER_KWH = "0.15"
python -m gridsearch.core --config ... --parallel 2
```

---

## 🔭 Análise Skyband (Multi-Critério)

O módulo inclui um motor de **Skyband Query** baseado em dominância de Pareto, que permite selecionar as melhores configurações em múltiplos critérios simultaneamente, com suporte a personalização de SLA por ambiente computacional.

### Conceitos Fundamentais

| Conceito | Definição |
|----------|-----------|
| **Dominância de Pareto** | `e_i` domina `e_j` se `e_i` é melhor ou igual em todos os critérios **e** estritamente melhor em pelo menos um |
| **Skyband k=1** | Frente de Pareto pura — pontos não dominados por nenhum outro |
| **Skyband_k** | Pontos dominados por **menos de k** outros — conjunto maior que a frente de Pareto |
| **Filtro de SLA** | Restricções de contexto aplicadas antes da dominância (ex: custo ≤ $5) |

### Parâmetros da Linha de Comando

```
python -m main [argumentos do grid] [argumentos Skyband]
```

#### Argumentos Skyband

| Argumento | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `--skyband` | flag | — | Executa análise Skyband **após** o grid search |
| `--skyband-only` | flag | — | **Apenas** análise Skyband sem executar novos experimentos |
| `--skyband-k K` | int | `1` | Ordem do Skyband. `k=1` = Pareto puro; `k=2` inclui segundo nível |
| `--sla-profile PERFIL` | str | — | Perfil predefinido (sobrescreve `--skyband-k`, `--skyband-metrics` e `--sla-constraint`) |
| `--sla-constraint M=V` | str (repetível) | — | Restrição de SLA no formato `metrica=valor_maximo` |
| `--skyband-metrics M…` | str+ | todos (5) | Lista de métricas para dominância de Pareto |
| `--skyband-compare` | flag | — | Exibe comparação Skyband vs ranking escalar (Jaccard + diferenças) |
| `--skyband-state ARQUIVO` | path | auto-detect | Caminho direto ao JSON de estado; default: arquivo mais recente |

#### Métricas disponíveis para `--skyband-metrics` (critérios de dominância Skyband)

```
train_time_sec    — tempo de treino em segundos
energy_kwh        — consumo energético em kWh
total_gflops      — custo computacional do modelo em GFLOPS
emissions_kg_co2  — emissões de CO₂ em kg (via CodeCarbon)
cost_usd          — custo estimado em USD
```

#### Métricas disponíveis para `--sla-constraint` (filtro de admissibilidade pré-dominância)

```
train_time_sec    — tempo de treino em segundos
energy_kwh        — consumo energético em kWh
peak_ram_mb       — pico de uso de RAM em MB  ← checagem de execução
emissions_kg_co2  — emissões de CO₂ em kg
cost_usd          — custo estimado em USD
```

#### Perfis de SLA Predefinidos (`--sla-profile`)

| Perfil | Constraints | k | Métricas usadas |
|--------|------------|---|-----------------|
| `economico` | `cost_usd ≤ 2.0` | 2 | custo, tempo, energia |
| `sustentavel` | `energy_kwh ≤ 0.05`, `emissions_kg_co2 ≤ 0.01` | 2 | energia, CO₂, custo |
| `tempo` | `train_time_sec ≤ 3600` | 3 | tempo, energia, custo |
| `balanceado` | `cost_usd ≤ 5.0`, `train_time_sec ≤ 7200`, `energy_kwh ≤ 0.1`, `emissions_kg_co2 ≤ 0.02` | 3 | tempo, custo, energia, CO₂ |
| `dev` | `train_time_sec ≤ 1800`, `peak_ram_mb ≤ 8192` | 2 | GFLOPs, tempo, energia |
| `producao` | `cost_usd ≤ 20.0`, `train_time_sec ≤ 1800`, `peak_ram_mb ≤ 16384` | 5 | tempo, GFLOPs, custo |

---

### Exemplos de Uso por Caso de Uso

#### 1. Frente de Pareto pura (Skyline k=1) sobre resultados existentes

```bash
# Carrega automaticamente o estado mais recente
python -m main --skyband-only
```

Saída esperada:

```
========================================================================
RELATÓRIO SKYBAND (k=1)
========================================================================
  SLA constraints : nenhuma
  Métricas        : ['train_time_sec', 'energy_kwh', 'total_gflops',
                     'emissions_kg_co2', 'cost_usd']
  Total de candidatos admissíveis : 3
  Tamanho do Skyband_1            : 1

Rank    Exp  Dom  Parâmetros                               Métricas
------------------------------------------------------------------------
0         2    0  {'optimizer': 'bert_adam', ...}   time=81.01  energy=0.00065  cost=7.76e-05
========================================================================
```

#### 2. Skyband k=2 com constraints de SLA customizadas

```bash
python -m main --skyband-only \
    --skyband-k 2 \
    --sla-constraint cost_usd=5.0 \
    --sla-constraint train_time_sec=7200
```

> Retorna os 2 melhores níveis de dominância entre os experimentos que custam ≤ $5 e treinam em ≤ 2h.

#### 3. Perfil de SLA predefinido (`balanceado`) + comparação vs escalar

```bash
python -m main --skyband-only --sla-profile balanceado --skyband-compare
```

Saída esperada (seção de comparação):

```
========================================================================
SKYBAND vs RANKING ESCALAR
========================================================================
  k                  : 3
  Jaccard similarity : 1.000
  Somente no Skyband : []
  Somente no Escalar : []
  Interseção         : [0, 1, 2]

  Skyband (preserva estrutura de dominância):
    rank=0 dom=0  bert_adam   train_time_sec=81.01  cost_usd=7.76e-05  energy_kwh=6.47e-04
    rank=1 dom=1  adam        train_time_sec=128.4  cost_usd=1.37e-04  energy_kwh=1.14e-03
    rank=2 dom=2  adamw       train_time_sec=129.6  cost_usd=1.40e-04  energy_kwh=1.17e-03

  Ranking Escalar (score ponderado min-max):
    [1] bert_adam   [2] adam   [3] adamw
========================================================================
```

> **Jaccard = 1.0** com 3 pontos indica concordância total. Com 216+ experimentos o Skyband revela trade-offs reais que o escalar colapsa.

#### 4. Skyband sobre arquivo de estado específico

```bash
python -m main --skyband-only \
    --skyband-state output/experiments/grid_search/grid_search_state_GPU_2026-03-01.json \
    --skyband-k 2 \
    --skyband-metrics train_time_sec cost_usd energy_kwh
```

#### 5. Apenas 2 critérios: tempo × custo

```bash
python -m main --skyband-only --skyband-metrics train_time_sec cost_usd
```

#### 6. Grid search completo com Skyband automático ao final

```bash
# Treina 216 configurações e logo após aplica Skyband com perfil sustentável
python -m main \
    --mode grid \
    --grid-config gridsearch/config/grid_search.json \
    --parallel 4 \
    --skyband \
    --skyband-k 3 \
    --sla-profile sustentavel \
    --skyband-compare
```

#### 7. Grid de teste (8 experimentos) + Skyband k=2 custom

```bash
python -m main \
    --mode grid \
    --grid-config gridsearch/config/grid_search_test.json \
    --parallel 2 \
    --skyband \
    --skyband-k 2 \
    --sla-constraint cost_usd=1.0 \
    --sla-constraint train_time_sec=3600 \
    --skyband-metrics train_time_sec cost_usd energy_kwh
```

---

### Uso Programático do Skyband

```python
import json
from pathlib import Path
from gridsearch.skyband import (
    skyband_query,
    pareto_front,
    sla_filter,
    compare_skyband_vs_ranking,
    skyband_report,
    DEFAULT_METRICS,
)

# Carrega e normaliza resultados do estado
state_file = sorted(
    Path("output/experiments/grid_search").glob("grid_search_state_*.json")
)[-1]
with open(state_file) as f:
    state = json.load(f)

results = [r for r in state["results"] if r.get("status") == "success"]
# Normaliza tipos (JSON pode armazenar numéricos como string)
for r in results:
    for k, v in r.get("resources", {}).items():
        if v is not None:
            r["resources"][k] = float(v)

# Frente de Pareto pura (k=1) — todos os 5 critérios
front = pareto_front(results)

# Skyband k=3 com SLA personalizada
recs = skyband_query(
    results,
    k=3,
    sla_constraints={"cost_usd": 5.0, "train_time_sec": 7200},
    metrics=["train_time_sec", "cost_usd", "energy_kwh"],
)
for r in recs:
    idx = r["grid_experiment_idx"]
    dom = r["domination_count"]
    params = r["grid_params"]
    print(f"Exp {idx:03d} | dom={dom} | {params}")

# Carrega perfil de SLA de arquivo JSON
with open("gridsearch/config/sla_profiles.json") as f:
    sla_cfg = json.load(f)
profile = sla_cfg["profiles"]["sustentavel"]
recs_sla = skyband_query(
    results,
    k=profile["skyband_k"],
    sla_constraints=profile["constraints"],
    metrics=profile["metrics"],
)

# Compara Skyband vs ranking escalar
report = compare_skyband_vs_ranking(
    results,
    sla={"cost_usd": 5.0},
    metrics=["train_time_sec", "cost_usd", "energy_kwh"],
    k=3,
)
print(f"Jaccard: {report['jaccard_similarity']:.2f}")
print(f"Somente no Skyband: {report['only_in_skyband']}")

# Relatório textual completo
print(skyband_report(results, k=2, sla_constraints={"cost_usd": 5.0}))
```

---

### Análise Manual

```python
from gridsearch.analysis import (
    compute_descriptive_statistics,
    analyze_correlations,
    rank_configurations
)

# Estatísticas descritivas de todas as métricas
stats = compute_descriptive_statistics(results)
print(f"Tempo médio: {stats['train_time']['mean']:.2f}s")
print(f"CO2 médio: {stats['emissions_kg_co2']['mean']:.6f} kg")
print(f"Custo médio: ${stats['cost_usd']['mean']:.4f}")

# Correlações entre hiperparâmetros e métricas
corr = analyze_correlations(results)

# Ranking multi-critério (exemplo: 40% tempo, 30% CO2, 30% custo)
top10 = rank_configurations(
    results,
    metrics=["train_time_sec", "emissions_kg_co2", "cost_usd"],
    weights=[0.4, 0.3, 0.3]
)[:10]

# Análise por hiperparâmetro específico
from gridsearch.analysis import analyze_by_hyperparameter

batch_impact_on_carbon = analyze_by_hyperparameter(
    results,
    param_name="batch_size",
    metric_name="emissions_kg_co2"
)

batch_impact_on_cost = analyze_by_hyperparameter(
    results,
    param_name="batch_size",
    metric_name="cost_usd"
)
```

## 🛡️ Validação de Memória

O módulo valida automaticamente disponibilidade de RAM antes da execução:

```python
from gridsearch.utils import check_memory_availability

is_safe, message = check_memory_availability(
    parallel_workers=2,
    max_batch_size=16
)

print(message)
```

Exemplo de saída:

```yaml
✓ Memória disponível: 23.9 GB
✓ Estimativa de uso: 7.5 GB
✓ Margem de segurança: 16.4 GB
✓ Sistema tem memória suficiente
```

## 🚨 Alertas Importantes

1. **Memória:** Com 32GB RAM, use `--parallel 2` no máximo
2. **Tempo:** Grid completo pode levar 3-4 dias
3. **Retomada:** Use `--resume` para continuar execuções interrompidas
4. **Backup:** Resultados parciais são salvos incrementalmente

## 📚 Documentação Completa

Consulte os arquivos em `gridsearch/docs/`:

- **GUIDE.md** - Guia detalhado de uso
- **QUICKSTART.md** - Tutorial de 5 minutos
- **OVERVIEW.md** - Visão técnica do módulo

## 🆘 Troubleshooting

### OOM (Out of Memory)

```powershell
# Reduza o paralelismo
python -m gridsearch.core --config ... --parallel 1
```

### Execução Interrompida

```powershell
# Retome de onde parou
python -m gridsearch.core --resume --parallel 2
```

### Resultados Corrompidos

```powershell
# Force nova execução
Remove-Item output/experiments/grid_search/grid_search_state.json
python -m gridsearch.core --config ... --parallel 2
```
