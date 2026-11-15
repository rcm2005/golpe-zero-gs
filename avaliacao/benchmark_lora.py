"""Comparativo de especializacao: LoRA vs RAG simples na mesma base 0.5B.

Avalia, sobre os 35 casos held-out (25 principais + 10 de estresse), tres
configuracoes do Qwen2.5-0.5B-Instruct:
  1. base, zero-shot (sem contexto);
  2. base + RAG simples (top-3 chunks oficiais no prompt);
  3. base + adapter LoRA (sem contexto, pesos especializados).
Metricas identicas as de ``avaliacao/benchmark.py``. Placar em
``avaliacao/placar_lora.json``.
"""

import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from avaliacao.benchmark import calcular_metricas, fontes_casam, normalizar_veredito
from notebooks.dataset_lora import INSTRUCAO
from notebooks.treinar_lora import MODELO_BASE, escolher_dispositivo
from src.llm_engine import SYSTEM_PROMPT_BASE
from src.vector_search import MotorBuscaVetorial

BASE_DIR = Path(__file__).resolve().parent.parent
ADAPTER_DIR = BASE_DIR / "adapters" / "golpe_zero_lora"
SAIDA = BASE_DIR / "avaliacao" / "placar_lora.json"
CHAVES = {"veredito", "tipo_golpe", "sinais_encontrados", "fonte_oficial", "proximo_passo"}


def extrair_json(texto: str) -> dict | None:
    for cand in (texto, *(re.findall(r"\{.*\}", texto, re.DOTALL))):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def gerar(modelo, tok, sistema: str, usuario: str, device: str) -> tuple[str, float]:
    msgs = [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = {k: v.to(device) for k, v in ids.items()}
    inicio = time.time()
    with torch.no_grad():
        out = modelo.generate(**ids, max_new_tokens=220, do_sample=False, pad_token_id=tok.pad_token_id)
    texto = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return texto, round(time.time() - inicio, 3)


def executar() -> dict:
    device = escolher_dispositivo()
    tok = AutoTokenizer.from_pretrained(MODELO_BASE)
    base = AutoModelForCausalLM.from_pretrained(MODELO_BASE, torch_dtype=torch.float32).to(device)
    modelo = PeftModel.from_pretrained(base, str(ADAPTER_DIR)).to(device)
    modelo.eval()
    motor = MotorBuscaVetorial()

    casos = []
    for split, nome in (("principal", "test_cases"), ("stress", "test_cases_stress")):
        with open(BASE_DIR / "avaliacao" / f"{nome}.json", encoding="utf-8") as f:
            casos += [{**c, "_split": split} for c in json.load(f)]
    with open(BASE_DIR / "base" / "chunks_oficiais.json", encoding="utf-8") as f:
        fontes = {c["id"]: c["fonte"] for c in json.load(f)}

    def contexto(msg: str) -> str:
        top = motor.buscar_top_k(msg, k=3)
        ctx = "\n\n".join(f"[{c['fonte']}]: {c['conteudo']}" for c in top)
        return (f"--- CONTEXTO OFICIAL DE SEGURANÇA ---\n{ctx}\n--- FIM DO CONTEXTO ---\n\n"
                f"MENSAGEM SUSPEITA RECEBIDA PELO IDOSO:\n\"{msg}\"\n\nDevolva o JSON conforme as instruções.")

    configs = {
        "Base 0.5B Zero-Shot": ("base", lambda m: (SYSTEM_PROMPT_BASE, f"Analise a seguinte mensagem suspeita: \"{m}\"")),
        "Base 0.5B + RAG Simples": ("base", lambda m: (SYSTEM_PROMPT_BASE, contexto(m))),
        "Base 0.5B + LoRA": ("lora", lambda m: (INSTRUCAO, m)),
    }
    placar = {}
    for nome, (modo, montar) in configs.items():
        print(f"\n[Avaliando] {nome}")
        detalhes = []
        for tc in casos:
            cat = tc["categoria_esperada"]
            esperado = "nao_consta" if cat == "fora_de_dominio" else cat
            sistema, usuario = montar(tc["mensagem_recebida"])
            if modo == "base":
                with modelo.disable_adapter():
                    texto, lat = gerar(modelo, tok, sistema, usuario, device)
            else:
                texto, lat = gerar(modelo, tok, sistema, usuario, device)
            obj = extrair_json(texto)
            conforme = obj is not None and CHAVES <= set(obj)
            obtido = normalizar_veredito((obj or {}).get("veredito", ""))
            fund = None
            if cat == "golpe":
                esp = [fontes[c] for c in tc.get("chunks_esperados", []) if c in fontes]
                fund = fontes_casam(str((obj or {}).get("fonte_oficial", "")), esp)
            detalhes.append({"id": tc["id"], "split": tc["_split"], "categoria_esperada": cat,
                             "veredito_esperado": esperado, "veredito_obtido": obtido,
                             "fonte_oficial": str((obj or {}).get("fonte_oficial", "")),
                             "fundamentacao": fund, "latencia_seg": lat, "conformidade_json": conforme})
            print(f"  [{tc['id']}] esperado={esperado:<10} obtido={obtido:<10} json={'ok' if conforme else 'FALHA'} {lat:.1f}s")
        placar[nome] = {
            **calcular_metricas(detalhes),
            "splits": {s: calcular_metricas([d for d in detalhes if d["split"] == s]) for s in ("principal", "stress")},
            "detalhes": detalhes,
        }

    print("\n" + "=" * 96)
    print(f"{'Configuração':<26} | {'Acc':>6} | {'Alucin.':>7} | {'JSON':>6} | {'Fundam.':>7} | {'Lat. média':>10}")
    for nome, m in placar.items():
        print(f"{nome:<26} | {m['acuracia_veredito_pct']:>5.1f}% | {m['taxa_alucinacao_pct']:>6.1f}% | "
              f"{m['conformidade_json_pct']:>5.1f}% | {m['taxa_fundamentacao_pct']:>6.1f}% | {m['latencia_media_seg']:>9.2f}s")
    SAIDA.write_text(json.dumps(placar, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPlacar salvo em {SAIDA}")
    return placar


if __name__ == "__main__":
    executar()
