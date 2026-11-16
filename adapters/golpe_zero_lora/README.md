# Adapter LoRA do Golpe Zero

- Modelo base: `Qwen/Qwen2.5-0.5B-Instruct`
- Configuração: r=8, alpha=16, dropout=0.05, módulos `q_proj` e `v_proj`
- Treino: 240 exemplos sintéticos (`avaliacao/dataset_treino_lora.jsonl`), 3 épocas, lr 0.0002
- Parâmetros treináveis: 540,672 de 494,573,440 (0.11%)
- Reprodução: `python -m notebooks.treinar_lora`; avaliação: `python -m avaliacao.benchmark_lora`
