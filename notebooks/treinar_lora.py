"""Treino LoRA real (PEFT) do Qwen2.5-0.5B-Instruct para o Golpe Zero.

Treina so os pesos LoRA (r=8, alpha=16, q_proj e v_proj). A perda e calculada
apenas sobre os tokens da resposta JSON. Salva o adapter em
``adapters/golpe_zero_lora/`` e o historico de loss em
``avaliacao/treino_lora_log.json``.
"""

import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from notebooks.dataset_lora import gerar_dataset, verificar_vazamento

BASE_DIR = Path(__file__).resolve().parent.parent
ADAPTER_DIR = BASE_DIR / "adapters" / "golpe_zero_lora"
LOG_PATH = BASE_DIR / "avaliacao" / "treino_lora_log.json"
MODELO_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
EPOCAS = 3
BATCH = 4
LR = 2e-4
MAX_LEN = 512
SEMENTE = 7


def escolher_dispositivo() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def montar_prompt(tok, instrucao: str, mensagem: str) -> str:
    msgs = [{"role": "system", "content": instrucao}, {"role": "user", "content": mensagem}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def tokenizar(tok, ex: dict[str, str]) -> dict[str, list[int]]:
    prompt = montar_prompt(tok, ex["instruction"], ex["input"])
    resposta = ex["output"] + "<|im_end|>"
    ids_prompt = tok(prompt, add_special_tokens=False)["input_ids"]
    ids_resp = tok(resposta, add_special_tokens=False)["input_ids"]
    ids = (ids_prompt + ids_resp)[:MAX_LEN]
    labels = ([-100] * len(ids_prompt) + ids_resp)[:MAX_LEN]
    return {"input_ids": ids, "labels": labels}


def lote(tok, itens: list[dict[str, list[int]]], device: str) -> dict[str, torch.Tensor]:
    tam = max(len(i["input_ids"]) for i in itens)
    pad = tok.pad_token_id
    ids = torch.tensor([i["input_ids"] + [pad] * (tam - len(i["input_ids"])) for i in itens])
    lab = torch.tensor([i["labels"] + [-100] * (tam - len(i["labels"])) for i in itens])
    att = torch.tensor([[1] * len(i["input_ids"]) + [0] * (tam - len(i["input_ids"])) for i in itens])
    return {"input_ids": ids.to(device), "labels": lab.to(device), "attention_mask": att.to(device)}


def executar_fine_tuning() -> dict:
    random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    device = escolher_dispositivo()
    dados = gerar_dataset(SEMENTE)
    assert verificar_vazamento(dados) == 0, "dataset de treino contem mensagens do gabarito"

    tok = AutoTokenizer.from_pretrained(MODELO_BASE)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO_BASE, torch_dtype=torch.float32).to(device)
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                     task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"])
    modelo = get_peft_model(modelo, cfg)
    treinaveis, total = modelo.get_nb_trainable_parameters()
    print(f"[LoRA] dispositivo={device} exemplos={len(dados)} treinaveis={treinaveis:,} de {total:,} ({treinaveis / total:.2%})")

    tokenizados = [tokenizar(tok, d) for d in dados]
    otim = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    passos_total = EPOCAS * ((len(tokenizados) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(otim, lambda s: max(0.05, 1 - s / passos_total))

    historico = []
    modelo.train()
    inicio = time.time()
    passo = 0
    for ep in range(1, EPOCAS + 1):
        random.shuffle(tokenizados)
        perdas = []
        for i in range(0, len(tokenizados), BATCH):
            out = modelo(**lote(tok, tokenizados[i:i + BATCH], device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otim.step(); sched.step(); otim.zero_grad()
            perdas.append(out.loss.item()); passo += 1
        media = sum(perdas) / len(perdas)
        historico.append({"epoca": ep, "loss_media": round(media, 4), "loss_final": round(perdas[-1], 4)})
        print(f"[LoRA] epoca {ep}/{EPOCAS} loss_media={media:.4f} ({time.time() - inicio:.0f}s)")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(str(ADAPTER_DIR))
    (ADAPTER_DIR / "README.md").write_text(
        "# Adapter LoRA do Golpe Zero\n\n"
        f"- Modelo base: `{MODELO_BASE}`\n"
        "- Configuração: r=8, alpha=16, dropout=0.05, módulos `q_proj` e `v_proj`\n"
        f"- Treino: {len(dados)} exemplos sintéticos (`avaliacao/dataset_treino_lora.jsonl`), {EPOCAS} épocas, lr {LR}\n"
        f"- Parâmetros treináveis: {treinaveis:,} de {total:,} ({treinaveis / total:.2%})\n"
        "- Reprodução: `python -m notebooks.treinar_lora`; avaliação: `python -m avaliacao.benchmark_lora`\n",
        encoding="utf-8",
    )
    with open(BASE_DIR / "avaliacao" / "dataset_treino_lora.jsonl", "w", encoding="utf-8") as f:
        for d in dados:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    resumo = {
        "modelo_base": MODELO_BASE, "dispositivo": device, "exemplos_treino": len(dados),
        "epocas": EPOCAS, "batch": BATCH, "lr": LR, "semente": SEMENTE,
        "parametros_treinaveis": treinaveis, "parametros_total": total,
        "tempo_treino_seg": round(time.time() - inicio, 1), "historico": historico,
    }
    LOG_PATH.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[LoRA] adapter salvo em {ADAPTER_DIR}")
    return resumo


if __name__ == "__main__":
    executar_fine_tuning()
