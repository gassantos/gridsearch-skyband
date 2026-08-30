# Workflow Generico

O comando abaixo executa uma DAG de tarefas para `ml_classic`,
`deep_learning`, `nlp` ou `llm`:

```powershell
uv run python -m main --workflow generic --workflow-spec workflow.json
```

Use `--workflow-dry-run` para validar dependencias, comandos e persistencia
do manifesto sem executar os comandos externos.

Ha exemplos autocontidos para `ml_classic`, `deep_learning`, `nlp` e `llm`
em `examples/workflow/`. Por exemplo:

```powershell
uv run python -m main --workflow generic --workflow-spec examples/workflow/nlp.json --workflow-dry-run
```

Para verificar a execucao ponta a ponta sem treinamento, execute o exemplo
autocontido `examples/workflow/smoke.json` sem `--workflow-dry-run`.

## Especificacao JSON

```json
{
  "name": "hf-text-classification",
  "experiment_type": "nlp",
  "monitoring": {
    "enable_emissions": true,
    "environment_cost_per_hour_usd": 1.2
  },
  "tasks": [
    {
      "task_id": "prepare_data",
      "name": "Preparar dataset",
      "task_type": "prepare",
      "command": ["python", "prepare.py"],
      "config": {"dataset": "glue", "subset": "mrpc"},
      "input_signatures": {"dataset": "glue-mrpc-v1"},
      "artifacts": {"dataset": "output/tokenized"}
    },
    {
      "task_id": "fine_tune",
      "name": "Fine-tuning",
      "task_type": "train",
      "depends_on": ["prepare_data"],
      "command": ["python", "train.py"],
      "metrics_file": "output/train_metrics.json"
    },
    {
      "task_id": "evaluate",
      "name": "Avaliar",
      "task_type": "evaluate",
      "depends_on": ["fine_tune"],
      "command": ["python", "evaluate.py"],
      "metrics_file": "output/evaluation_metrics.json"
    }
  ]
}
```

Cada comando deve retornar codigo zero. Um `metrics_file`, quando declarado,
deve conter um objeto JSON compativel com `TaskExecutionAttempt.metrics`, por
exemplo `{"evaluation": {"f1_score": 0.91}}` ou
`{"resources": {"total_gflops": 123.4}}`. A instrumentacao do executor
acrescenta tempo, RAM, VRAM, energia, emissoes e custo por tentativa.

O workflow generico orquestra ferramentas externas; ele nao impoe uma
biblioteca de treinamento. Assim, o mesmo contrato atende scikit-learn,
PyTorch, TensorFlow e Hugging Face Transformers.

## Etapas Base Por Dominio

Antes de associar comandos ou bibliotecas, use `build_domain_workflow` para
criar a DAG canonica do tipo de experimento. A fonte dos dados e o modelo sao
perfis de tarefa, nao parte da definicao das etapas:

| Tipo | Etapas basicas |
| --- | --- |
| `ml_classic` | carregar dados, preparar atributos, treinar, avaliar |
| `deep_learning` | carregar dados, preparar dados, treinar, validar, avaliar |
| `nlp` | carregar textos, preprocessar textos, treinar, avaliar |
| `llm` | carregar dados, preparar corpus, adaptar, avaliar, publicar |

Os pipelines concretos serao associados posteriormente: `scikit-learn`
Pipeline para ML classico, PyTorch para Deep Learning, Hugging Face para NLP,
e uma biblioteca de orquestracao apropriada para LLM.
