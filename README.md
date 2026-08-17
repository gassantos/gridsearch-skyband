# PSLA4ML: SLAs Personalizados via 𝑘-Skyband para para Treino ML

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.9.0-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/transformers-%E2%89%A55.2.0-FFD21E?logo=huggingface&logoColor=black)
![codecarbon](https://img.shields.io/badge/codecarbon-%E2%89%A53.2.2-4CAF50?logo=leaflet&logoColor=white)
![Tests](https://img.shields.io/badge/tests-400%2B%20passing-brightgreen?logo=pytest&logoColor=white)

## Instalação

O projeto usa [`uv`](https://docs.astral.sh/uv/) como gerenciador de ambiente e dependências. O `requirements.txt` presente no repositório é gerado automaticamente — **não o edite diretamente**; o referencial é o `pyproject.toml`.

### Pré-requisitos

- Python ≥ 3.12, < 3.14
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) instalado
- (Opcional) NVIDIA GPU com CUDA 12.8 para aceleração

### Passos

```bash
# 1. Clone o repositório
git clone https://github.com/gassantos/gridsearch-skyband.git
cd gridsearch-skyband

# 2. Crie o ambiente virtual e instale as dependências de produção
uv sync

# 3. (Opcional) Instale também as dependências de desenvolvimento (pytest, etc.)
uv sync --group dev
```

> **Plataforma Linux (CUDA 12.8):** o `pyproject.toml` já configura automaticamente o índice `pytorch-cu128`. Em macOS/Windows, o índice `pytorch-cpu` é selecionado sem necessidade de configuração adicional.

### Verificar instalação

```bash
uv run python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

---

## Testes

A suite de testes cobre os componentes críticos do pipeline de treinamento (mais de 400 testes pytest).

```bash
# Suite completa
uv run --group dev pytest

# Com relatório de cobertura por módulo
uv run --group dev pytest --cov=. --cov-report=term-missing

# Arquivo específico
uv run --group dev pytest tests/test_optimizer.py -v

# Classe específica
uv run --group dev pytest tests/test_warmup_scheduler.py::TestSchedulerLRBehavior -v
```

| Arquivo de teste | Cobertura |
| --- | --- |
| `test_optimizer.py` | Tipos e hiperparâmetros de todos os otimizadores (Adam, AdamW, SGD, bert_adam) |
| `test_warmup_scheduler.py` | Cálculo de steps, comportamento de LR no warmup/decaimento, restauração de estado |
| `test_checkpoint.py` | Chaves obrigatórias, round-trip do warmup scheduler, `trained_epoch` e `global_step` |
| `test_init_tool_state.py` | Carregamento de estado em `init_all`, tolerância a checkpoint inválido |
| `test_gridsearch.py` | Geração de grade, validação de memória, filtragem de config, análise de resultados |
| `test_gridsearch_config.py` | Estrutura e naming dos arquivos JSON de configuração do grid |
| `test_skyband.py` | Dominância Pareto/Skyband, filtro de SLA, comparação vs. ranking escalar |
| `test_psla4ml.py` | Discretização de métricas, `Tier`, `TrainingTemplate`, geração PSLA4ML |
| `test_eval_metrics.py` | Cálculo de precision/recall/F1/accuracy na avaliação |
| `test_device.py` | Detecção e seleção de dispositivo (CPU/CUDA/MPS) |
| `test_huggingface_dataset.py` | Carregamento via HuggingFace Hub e JSONL local |
| `test_tpu_check.py` | Checagem de disponibilidade de TPU |
| `test_run_experiment_fixes.py` | Regressões do motor de execução central (`execute_experiment`) |

---

## Pipeline de Funcionamento

<img width="934" height="370" alt="PSLA4ML-SBBD" src="https://github.com/user-attachments/assets/27edddd7-9c39-4d85-853a-9353e59a30d6" />

---

## Executando Experimentos Rastreáveis

Para pesquisa e reprodutibilidade, use `main.py` como ponto de entrada único. Ele orquestra tanto experimentos individuais quanto busca em grade de hiperparâmetros, delegando a execução ao pacote `experiment/`.

### Arquitetura de execução

```py
main.py  ──(mode=single)──→  experiment/  →  tools/train_tool.py
         ──(mode=grid)────→  gridsearch/core.py  →  experiment/
```

### Experimento Single-mode

```bash
# Execução padrão (usa config/experiments/BertPLI.config)
uv run python -m main --mode single

# Com configuração específica
uv run python -m main --mode single --config config/experiments/BertPLI2.config

# Com seleção explícita de GPU
uv run python -m main --mode single --gpu 0          # GPU 0
uv run python -m main --mode single --gpu 0 1        # GPU 0+1
```
---

Este repositório faz referência ao BERT-PLI, publicado no artigo da IJCAI-PRICAI 2020: *BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval*.

### Experimento Grid-mode

```bash
# Busca minimal (teste rápido — 8 combinações, 2 workers)
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search_test.json \
    --parallel 2

# Busca completa (produção — 4.860 combinações; use --sla-constraint/--sla-profile para pré-filtrar)
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search.json \
    --parallel 4

# Busca multiambiente (9.720 combinações = hiperparâmetros x 5 ambientes ativos)
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search_multienv.json \
    --parallel 2

# Retomar execução interrompida
uv run python -m main --mode grid --resume

# Aplicar SLA profile também como pré-filtro de execução
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search.json \
    --parallel 4 \
    --sla-profile dev
```

> **Distribuição de GPUs:** em modo paralelo, o `main.py` distribui os workers em round-robin pelas GPUs disponíveis de forma automática. Para controle explícito, use `--gpu 0 1`.
> **Pré-filtro SLA:** em modo `grid`, `--sla-profile` e `--sla-constraint` agora também filtram combinações antes da execução quando a constraint é estimável. Hoje isso vale diretamente para `peak_ram_mb` e para `train_time_sec` quando o JSON da grade expõe uma baseline de tempo em `_meta.time_estimation` ou no fallback `_meta.per_experiment_train_time_sec`. No grid multiambiente, a estimativa pode usar o baseline específico de cada ambiente.
> **Grid multiambiente:** quando `environments.active` está presente no JSON, o grid executa o produto `hyperparameters × environments`, adicionando o campo de ambiente aos parâmetros de cada experimento.

### TPU multicore no Google Colab

Selecione um runtime TPU no Colab e execute, a partir do clone do repositório:

```bash
pip install uv
uv sync --extra tpu --group dev
export PJRT_DEVICE=TPU

uv run python -m main --mode single \
    --config config/experiments/BertPLI.config \
    --precision bf16 \
    --tpu-cores 8 \
    --no-skyband

uv run python -m scripts.homologate_tpu --expected-cores 8
```

O override `--precision bf16` é obrigatório nesse exemplo porque o baseline `BertPLI.config` usa FP16, formato não suportado por este pipeline em XLA. O último comando homologa o BL-08 somente quando o experimento termina com sucesso, registra `device_type=TPU`, usa os oito workers PJRT e contém contagens positivas de `CompileTime` e `ExecuteTime`. Em grid TPU multicore, use `--parallel 1`; o paralelismo ocorre entre os cores TPU dentro de cada experimento.

### Dataset via HuggingFace Hub ou JSONL local

Além dos datasets em `data/` referenciados pelo `.config`, o CLI aceita fontes alternativas via `--dataset-source`, sobrescrevendo o `DataLoader` configurado:

```bash
# Dataset público do HuggingFace Hub (ex.: glue/mrpc)
uv run python -m main --mode single \
    --dataset-source hub --dataset-id nyu-mll/glue --dataset-config mrpc

# Dataset local em JSONL via HuggingFace Datasets
uv run python -m main --mode single --dataset-source local_json

# Grid search combinando dataset do Hub com pré-filtro de SLA
uv run python -m main --mode grid --sla-profile dev \
    --dataset-source hub --dataset-id nyu-mll/glue --dataset-config mrpc
```

> Implementado em [dataset/nlp/HuggingFace.py](dataset/nlp/HuggingFace.py). `--dataset-source` aceita `hub` (requer `--dataset-id`; `--dataset-config` é opcional) ou `local_json` (usa arquivos JSONL locais, ignorando o `DataLoader` do `.config`).

---

## Análise Skyband e Perfis de SLA

Ao final de cada execução (`single` ou `grid`), o CLI roda automaticamente uma análise **Skyband** (dominância Pareto com tolerância `k`) sobre as métricas de recursos e qualidade, ranqueando as configurações não-dominadas. O comportamento é implementado em [gridsearch/dominance.py](gridsearch/dominance.py), [gridsearch/comparison.py](gridsearch/comparison.py), [gridsearch/tiers.py](gridsearch/tiers.py) e [gridsearch/visualization.py](gridsearch/visualization.py), orquestrado por [cli/runners.py](cli/runners.py).

```bash
# Desativar a análise Skyband automática (executa somente os experimentos)
uv run python -m main --mode grid --no-skyband

# Analisar um estado de grid search já existente, sem novo treino
uv run python -m main --skyband-only

# Skyband-only com k customizado e perfil de SLA sustentável
uv run python -m main --skyband-only --skyband-k 2 --sla-profile sustentavel

# Skyband-only com constraints de SLA customizadas (pode repetir a flag)
uv run python -m main --skyband-only \
    --sla-constraint cost_usd=5.0 \
    --sla-constraint train_time_sec=7200

# Comparar o resultado do Skyband contra um ranking escalar tradicional
uv run python -m main --skyband-only --sla-profile balanceado --skyband-compare

# Analisar um arquivo de estado específico com métricas customizadas
uv run python -m main --skyband-only \
    --skyband-state output/experiments/grid_search/grid_search_state_GPU_2026-03-01.json \
    --skyband-k 2 --skyband-metrics train_time_sec cost_usd energy_kwh
```

### Perfis de SLA disponíveis (`--sla-profile`)

| Perfil | Restrições |
| --- | --- |
| `economico` | `cost_usd <= 2.00` |
| `sustentavel` | `energy_kwh <= 0.05`, `emissions_kg_co2 <= 0.01` |
| `tempo` | `train_time_sec <= 3600` |
| `balanceado` | `cost_usd <= 5.00`, `train_time_sec <= 7200`, `energy_kwh <= 0.1` |
| `dev` | `train_time_sec <= 1800`, `peak_ram_mb <= 8192` |
| `producao` | `cost_usd <= 20.00`, `train_time_sec <= 1800`, `peak_ram_mb <= 16384` |

Perfis definidos em `gridsearch/config/sla_profiles.json`. Métricas aceitas em `--sla-constraint` (filtro de admissibilidade): `train_time_sec`, `energy_kwh`, `peak_ram_mb`, `emissions_kg_co2`, `cost_usd`. Métricas aceitas em `--skyband-metrics` (critérios de dominância): `train_time_sec`, `energy_kwh`, `total_gflops`, `emissions_kg_co2`, `cost_usd`.

### Discretização PSLA4ML (Tiers)

O módulo [gridsearch/tiers.py](gridsearch/tiers.py) implementa a discretização de métricas em faixas (`Tier`) e a geração de `TrainingTemplate`s a partir do conjunto Skyband, permitindo filtrar configurações por template (`filter_by_template`) além do ranking bruto.

### Análises estatísticas complementares

O pacote [gridsearch/analysis/](gridsearch/analysis) fornece funções de apoio à interpretação dos resultados do grid search: correlações entre hiperparâmetros e métricas (`correlations.py`), estatísticas descritivas (`statistics.py`), ranking configurável (`ranking.py`) e geração de relatórios (`report.py`).

### Artefatos gerados

Cada experimento produz automaticamente em `output/experiments/metrics/`:

| Artefato | Descrição |
| --- | --- |
| `<nome>_<optimizer>_<lr>_<bs>_<ep>_<timestamp>.json` | Métricas completas do experimento em JSON |
| `experiment_summary_<YYYYMMDD>.csv` | Agregação diária de todos os experimentos |
| `EmissionsCO2_<device>_<YYYYMMDD>.csv` | Emissões de CO₂ rastreadas pelo `codecarbon` |

O JSON por experimento contém as seções:

```json
{
  "experiment":      { "id", "config_name", "seed", "status", "timestamp_start", "timestamp_end" },
  "environment":     { "device_type", "device_name", "precision" },
  "hyperparameters": { "optimizer", "learning_rate", "batch_size", "epoch", "avg_gflops_per_batch" },
  "resources":       { "train_time_sec", "energy_kwh", "emissions_kg_co2", "avg_ram_mb", "peak_ram_mb", "total_gflops" },
  "evaluation":      { "precision", "recall", "f1_score", "source" },
  "logs":            { "stdout_tail", "stderr_tail" }
}
```


### Rastreamento de Emissões de CO₂

O projeto integra o [`codecarbon`](https://mlco2.github.io/codecarbon/) para medir consumo
energético e emissões de CO₂ a cada experimento executado.

**Ativar no arquivo `.config` do experimento:**

```ini
[monitoring]
enable_monitoring = true
```

Quando ativado, ao término do experimento:

| Saída | Descrição |
| ------- | ----------- |
| Campo `energy_kwh` no JSON | Energia consumida em kWh |
| Campo `emissions_kg_co2` no JSON | Emissões estimadas em kg CO₂ |
| `output/experiments/metrics/EmissionsCO2_<device>_<YYYYMMDD>.csv` | Histórico acumulado de emissões |

### **Custo estimado de energia**

O custo monetário é calculado com a tarifa padrão de **$0,12 USD/kWh**, configurável via variável de ambiente:

```bash
ENERGY_COST_USD_PER_KWH=0.08 uv run python -m main --mode single
```

> A variável `ENERGY_COST_USD_PER_KWH` aceita qualquer valor em ponto flutuante (USD por kWh).
> O resultado aparece no campo `energy_cost_usd` do JSON de métricas do experimento.

---

## Busca em Grade de Hiperparâmetros (Grid Search)

O módulo `gridsearch/` implementa busca exaustiva de hiperparâmetros com execução paralela, rastreamento de recursos e análise automática dos resultados.

### Modos de execução

**Via `main.py` (recomendado):**

```bash
# Grade de teste — 8 combinações, validação rápida do pipeline (~2-3h com 2 workers)
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search_test.json \
    --parallel 2

# Grade completa — 4.860 combinações (tempo total varia com hardware; baseline de ~1800s/exp em config['_meta'])
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search.json \
    --parallel 4

# Retomar execução interrompida (checkpoint automático por experimento)
uv run python -m main --mode grid --resume
```

**Via módulo `gridsearch` (uso programático):**

```python
from gridsearch import run_grid_search, analyze_results
import json

with open("gridsearch/config/grid_search_test.json") as f:
    grid_config = json.load(f)

results = run_grid_search(
    base_config_path="config/experiments/BertPLI.config",
    grid_config=grid_config["hyperparameters"],
    parallel=2,
    gpu_ids=[0, 1],   # distribuição round-robin por worker
)

analysis = analyze_results(results)
```

### Espaço de busca

| Hiperparâmetro | Valores (grade completa) |
| --- | --- |
| `learning_rate` | `5e-6`, `1e-5`, `2e-5`, `3e-5`, `5e-5` |
| `batch_size` | `8`, `16`, `32` |
| `optimizer` | `adam`, `adamw`, `sgd`, `bert_adam` |
| `dropout` | `0.1`, `0.2`, `0.3` |
| `seed` | `42`, `123`, `456` |
| `max_seq_length` | `128`, `256`, `512` |
| `num_epochs` | `2`, `3`, `5` |
| **Total** | **4.860 combinações** |

> Para explorar apenas o impacto na qualidade preditiva com menor custo, prefira `gridsearch/config/grid_search_quality.json` (grade reduzida).

Artefatos gerados

```sh
output/experiments/grid_search/
├── grid_search_results_<data>.json    # Resultados de todos os experimentos
├── grid_search_summary_<data>.txt     # Ranking legível das melhores configurações
└── grid_search_state_<data>.json      # Estado para retomada (--resume)
```

O módulo analisa e ranqueia as configurações por 5 critérios:
tempo de treinamento, consumo de energia (kWh), emissões de CO₂ (kg),
uso de RAM (MB) e F1-score de validação.

> Para documentação completa do módulo, consulte [`gridsearch/README.md`](gridsearch/README.md).

---

## Estrutura do Projeto

```md
📁 Gridsearch-Skyband/
│
├── 📄 main.py                 Orquestrador do experimento
├── 📄 pyproject.toml          Dependências e entrypoints
├── 📄 compose.yaml            Ambiente containerizado
├── 📄 Dockerfile              Imagem base do projeto
├── 📁 cli/                    CLI (Command Pattern): parser, comandos e runners
├── 📁 config/                 Configurações em cascata
├── 📁 model/                  Modelos LM
├── 📁 formatter/               Preparação de inputs
├── 📁 dataset/                DataLoaders (locais e HuggingFace Hub)
├── 📁 tools/                  Treino, Avaliação e Inferência
├── 📁 scripts/                Entrypoints CLI
├── 📁 gridsearch/             Módulo de busca em grid, Skyband e análise de SLA
│   ├── core.py / grid.py / executor.py / protocols.py   Motor de grid search
│   ├── dominance.py / comparison.py / visualization.py  Pareto / Skyband
│   ├── tiers.py                                         Discretização PSLA4ML
│   ├── sla_prefilter.py                                 Perfis e pré-filtro de SLA
│   └── analysis/                                        Correlações, estatísticas e ranking
├── 📁 utils/                  Utilitários gerais
├── 📁 tests/                  Suite com 400+ testes
├── 📁 data/                   Dados sintéticos
├── 📁 examples/               Exemplos de dados
├── 📁 docs/                   Documentação técnica
└── 📁 devconteiner/           Dev Container (VS Code)
```

### Exemplos de formato de dados

> **Para fins acadêmicos:** Este repositório inclui dados sintéticos de casos jurídicos no diretório `data/`, simulando a estrutura do dataset COLIEE:
>
> - 34 pares de parágrafos para treino (exemplos positivos e negativos balanceados)
> - 6 pares de parágrafos para validação
> - 10 documentos com múltiplos parágrafos para teste
> - Conteúdo jurídico realista cobrindo diversos temas (contratos, direito constitucional, processo civil, etc.)
>
> **Para pesquisa/produção:** acesse [COLIEE 2019](https://sites.ualberta.ca/~rabelo/COLIEE2019/) para solicitar o dataset original da competição.

- [examples/task2/data_sample.json](examples/task2/data_sample.json): input para Estágio 2 (fine-tuning par de parágrafos)

```json
{ "guid": "queryID_paraID", "text_a": "<parágrafo decisão>", "text_b": "<parágrafo candidato>", "label": 0 }
```

- [examples/task1/case_para_sample.json](examples/task1/case_para_sample.json): input para Estágio 3 (BertPoolOutMax)

```json
{ "guid": "queryID_docID", "q_paras": ["..."], "c_paras": ["..."], "label": 0 }
```

- [examples/task1/embedding_sample.json](examples/task1/embedding_sample.json): input para Estágio 4 (AttenRNN)

```json
{ "guid": "queryID_docID", "res": [[...], ...], "label": 0 }
```

### Dependências

- Gerenciadas via `pyproject.toml` + `uv`. Consulte a seção [Instalação](#instalação) para instruções completas.
- Para inspecionar as versões exatas resolvidas: `uv pip list`

## Reprodutibilidade

O projeto garante resultados reproduzíveis por meio da função `set_seed` em [utils/seed.py](utils/seed.py):

| Camada | Mecanismo |
| -------- | ----------- |
| Python | `random.seed(seed)` |
| Python hash | `PYTHONHASHSEED=<seed>` |
| NumPy | `np.random.seed(seed)` |
| PyTorch CPU | `torch.manual_seed(seed)` |
| PyTorch CUDA | `torch.cuda.manual_seed_all(seed)` |
| cuDNN | `deterministic=True`, `benchmark=False` |
| Apple Silicon (MPS) | `torch.mps.manual_seed(seed)` |
| Transformers | `transformers.set_seed(seed)` |

### Configurando o seed pelo arquivo `.config`

```ini
[training]
seed = 42
```

O seed padrão é `42`. Para máxima reprodutibilidade, use `ensure_reproducibility()`, que adicionalmente define `CUBLAS_WORKSPACE_CONFIG=:4096:8`:

```python
from utils.seed import ensure_reproducibility
ensure_reproducibility(seed=42)
```

### Trade-off: determinismo vs. performance

| Modo | `cudnn.deterministic` | `cudnn.benchmark` | Performance |
| ------ | ---------------------- | ------------------- | ------------- |
| `set_seed(seed, deterministic=True)` *(padrão)* | `True` | `False` | Reduzida |
| `set_seed(seed, deterministic=False)` | `False` | `True` | Máxima |

> **Nota:** mesmo com `deterministic=True`, operações atômicas em GPU (ex.: `scatter_add`) podem introduzir variação residual em versões mais antigas do CUDA. Para eliminação total, use `ensure_reproducibility()`.

---
