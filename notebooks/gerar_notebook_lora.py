"""Gera e executa o notebook 02_fine_tuning_lora.ipynb com saidas salvas.

O notebook chama os mesmos modulos do repositorio (dataset, treino e benchmark),
entao executa o treino LoRA real e a avaliacao held-out. Ao final, uma celula
de conclusao e adicionada com os numeros lidos de ``avaliacao/placar_lora.json``.

Uso: python -m notebooks.gerar_notebook_lora
"""

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

BASE_DIR = Path(__file__).resolve().parent.parent
NB_PATH = BASE_DIR / "notebooks" / "02_fine_tuning_lora.ipynb"


def md(texto: str):
    return nbformat.v4.new_markdown_cell(texto)


def code(texto: str):
    return nbformat.v4.new_code_cell(texto)


def construir() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md("# FIAP Global Solution 2026.2 • CitizenTech\n"
           "## Golpe Zero — Especialização com LoRA (PEFT) vs RAG simples\n"
           "**Disciplina:** Generative AI for Engineering · **Professor:** Francisco Elânio Bezerra\n\n"
           "Este caderno treina um adapter LoRA de verdade sobre o `Qwen2.5-0.5B-Instruct` e compara, nos 35 casos held-out, "
           "a especialização paramétrica (LoRA) com a especialização contextual (RAG simples) na mesma base de 0.5B.\n\n"
           "1. Fundamentação do LoRA · 2. Dataset de treino · 3. Hiperparâmetros · 4. Treino · 5. Adapter exportado · "
           "6. Placar LoRA vs RAG simples · 7. Conclusão"),
        md("### 1. Fundamentação do LoRA\n"
           "Em vez de atualizar a matriz de pesos $W \\in \\mathbb{R}^{d \\times k}$, o LoRA congela $W$ e aprende uma atualização de posto baixo:\n\n"
           "$$W' = W + \\frac{\\alpha}{r} B A, \\quad B \\in \\mathbb{R}^{d \\times r},\; A \\in \\mathbb{R}^{r \\times k},\; r \\ll \\min(d, k)$$\n\n"
           "Com $r=8$ e $\\alpha=16$ aplicados a `q_proj` e `v_proj`, treina-se uma fração mínima dos parâmetros."),
        md("### 2. Dataset de treino (sintético, disjunto do gabarito)"),
        code("import json\nfrom notebooks.dataset_lora import gerar_dataset, verificar_vazamento\n\n"
             "dados = gerar_dataset(7)\nfrom collections import Counter\n"
             "print('exemplos:', len(dados))\n"
             "print('mensagens de treino idênticas ao gabarito:', verificar_vazamento(dados))\n"
             "print('distribuição de vereditos:', Counter(json.loads(d['output'])['veredito'] for d in dados))\n"
             "print('\\nExemplo de treino:')\nprint(dados[0]['input'])\nprint(dados[0]['output'])"),
        md("### 3. Hiperparâmetros LoRA"),
        code("from peft import LoraConfig, TaskType\n\n"
             "cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias='none',\n"
             "                 task_type=TaskType.CAUSAL_LM, target_modules=['q_proj', 'v_proj'])\nprint(cfg)"),
        md("### 4. Treino real (perda só nos tokens da resposta JSON)"),
        code("from notebooks.treinar_lora import executar_fine_tuning\n\nresumo = executar_fine_tuning()\n"
             "print(json.dumps(resumo, indent=2, ensure_ascii=False))"),
        md("### 5. Adapter exportado"),
        code("from pathlib import Path\n\n"
             "for arq in sorted(Path('adapters/golpe_zero_lora').glob('*')):\n"
             "    print(f'{arq.name:<30} {arq.stat().st_size/1024:>9.1f} KB')\n"
             "cfg_salvo = json.load(open('adapters/golpe_zero_lora/adapter_config.json'))\n"
             "print('base_model_name_or_path:', cfg_salvo['base_model_name_or_path'])"),
        md("### 6. Placar held-out: base zero-shot · base + RAG simples · base + LoRA\n"
           "Os 35 casos (25 principais + 10 de estresse) não foram vistos no treino."),
        code("from avaliacao.benchmark_lora import executar\n\nplacar = executar()"),
    ]
    return nb


def conclusao(placar: dict) -> str:
    z, r, l = (placar["Base 0.5B Zero-Shot"], placar["Base 0.5B + RAG Simples"], placar["Base 0.5B + LoRA"])
    return (
        "### 7. Conclusão\n\n"
        "| Configuração | Acurácia | Alucinação | JSON válido | Fundamentação | Latência média |\n| :--- | :---: | :---: | :---: | :---: | :---: |\n"
        + "\n".join(
            f"| {n} | {m['acuracia_veredito_pct']}% | {m['taxa_alucinacao_pct']}% | {m['conformidade_json_pct']}% | "
            f"{m['taxa_fundamentacao_pct']}% | {m['latencia_media_seg']} s |"
            for n, m in (("Base 0.5B zero-shot", z), ("Base 0.5B + RAG simples", r), ("Base 0.5B + LoRA", l))
        )
        + "\n\nInterpretação detalhada em `relatorio/05_lora.md`."
    )


def escrever_relatorio(placar: dict, log: dict) -> None:
    """Gera relatorio/05_lora.md a partir dos numeros medidos."""
    z, r, l = (placar["Base 0.5B Zero-Shot"], placar["Base 0.5B + RAG Simples"], placar["Base 0.5B + LoRA"])

    def linha(n, m):
        return (f"| {n} | {m['acuracia_veredito_pct']}% | {m['taxa_alucinacao_pct']}% | {m['conformidade_json_pct']}% | "
                f"{m['taxa_fundamentacao_pct']}% | {m['splits']['stress']['acuracia_veredito_pct']}% | {m['latencia_media_seg']} s |")

    def melhor(chave, maior=True):
        a, b = l[chave], r[chave]
        if a == b:
            return "empate"
        return "LoRA" if (a > b) == maior else "RAG simples"

    tabela = "\n".join(linha(n, m) for n, m in (("Base 0.5B zero-shot", z), ("Base 0.5B + RAG simples", r), ("Base 0.5B + LoRA", l)))
    texto = f"""# Especialização: LoRA vs RAG simples (base Qwen2.5-0.5B-Instruct)

## Protocolo
- Modelo base: `{log['modelo_base']}`. Mesma base nas três configurações; o adapter é desligado nas duas primeiras.
- Treino LoRA: {log['exemplos_treino']} exemplos sintéticos (`avaliacao/dataset_treino_lora.jsonl`), r=8, alpha=16, módulos `q_proj` e `v_proj`, {log['epocas']} épocas, lr {log['lr']}, {log['parametros_treinaveis']:,} parâmetros treináveis de {log['parametros_total']:,} ({log['parametros_treinaveis'] / log['parametros_total']:.2%}).
- Avaliação held-out: os 35 casos de `avaliacao/test_cases.json` e `avaliacao/test_cases_stress.json`. Nenhuma mensagem do gabarito foi usada no treino (verificado em código).
- Métricas idênticas às do benchmark principal. JSON válido exige as 5 chaves do schema.

## Resultado
| Configuração | Acurácia | Alucinação | JSON válido | Fundamentação | Acurácia estresse | Latência média |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{tabela}

## Interpretação
- Acurácia de veredito: {melhor('acuracia_veredito_pct')} à frente na comparação LoRA vs RAG simples ({l['acuracia_veredito_pct']}% contra {r['acuracia_veredito_pct']}%).
- Alucinação (menor é melhor): {melhor('taxa_alucinacao_pct', maior=False)} ({l['taxa_alucinacao_pct']}% contra {r['taxa_alucinacao_pct']}%).
- JSON válido: {melhor('conformidade_json_pct')} ({l['conformidade_json_pct']}% contra {r['conformidade_json_pct']}%).
- Fundamentação em fonte oficial: {melhor('taxa_fundamentacao_pct')} ({l['taxa_fundamentacao_pct']}% contra {r['taxa_fundamentacao_pct']}%). O RAG injeta o texto oficial no prompt; o LoRA só aprende o padrão de citação do treino.
- Latência: o LoRA não carrega contexto no prompt ({l['latencia_media_seg']} s contra {r['latencia_media_seg']} s do RAG simples).
- Limites: 35 casos, dataset de treino sintético e modelo de 0.5B. Diferenças de poucos pontos equivalem a 1 ou 2 casos e não são conclusivas. O LoRA não substitui a base oficial: ele não atualiza conhecimento, só o formato e o critério de decisão.
"""
    (BASE_DIR / "relatorio" / "05_lora.md").write_text(texto, encoding="utf-8")


def main() -> None:
    nb = construir()
    NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(BASE_DIR)}}).execute()
    placar = json.loads((BASE_DIR / "avaliacao" / "placar_lora.json").read_text(encoding="utf-8"))
    nb.cells.append(md(conclusao(placar)))
    escrever_relatorio(placar, json.loads((BASE_DIR / "avaliacao" / "treino_lora_log.json").read_text(encoding="utf-8")))
    nbformat.write(nb, NB_PATH)
    print(f"notebook salvo com saídas em {NB_PATH}")


if __name__ == "__main__":
    main()
