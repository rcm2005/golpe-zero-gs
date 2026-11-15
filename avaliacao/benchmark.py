"""Script de Benchmark Comparativo das 3 Arquiteturas (Fase 1.4).

Executa a avaliação sistemática sobre os 35 casos de teste (25 principais
de ``test_cases.json`` + 10 de estresse de ``test_cases_stress.json``) e gera
o placar comparativo oficial com:
- Acurácia de veredito, taxa de alucinação (falso positivo), conformidade JSON;
- Latência média global e latência média em casos fora de domínio;
- Taxa de fundamentação: nos casos de golpe, fração em que a fonte citada em
  ``fonte_oficial`` casa (normalização: minúsculas, sem acento/pontuação,
  substring em qualquer direção) com a fonte de algum chunk do gabarito
  ``chunks_esperados``;
- Breakdown por split (principal / stress) e detalhes COMPLETOS por caso.

O placar é persistido em ``avaliacao/placar_benchmark.json``.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from src.llm_engine import PipelineGolpeZero

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_CASES_PATH = BASE_DIR / "avaliacao" / "test_cases.json"
TEST_CASES_STRESS_PATH = BASE_DIR / "avaliacao" / "test_cases_stress.json"
CHUNKS_PATH = BASE_DIR / "base" / "chunks_oficiais.json"
RELATORIO_OUTPUT = BASE_DIR / "avaliacao" / "placar_benchmark.json"

SPLITS = ("principal", "stress")


def normalizar_veredito(veredito_raw: str) -> str:
    """Normaliza o veredito retornado pelo modelo para comparação estrita."""
    v = str(veredito_raw).strip().lower()
    if "golpe" in v:
        return "golpe"
    if "legitim" in v:
        return "legitimo"
    if "nao_consta" in v or "não consta" in v or "desconhecid" in v:
        return "nao_consta"
    return "outro"


def normalizar_fonte(texto: str) -> str:
    """Normalização de string para comparação de fontes:
    minúsculas, remoção de acentos (NFKD) e de pontuação, espaços colapsados.
    """
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def fontes_casam(fonte_citada: str, fontes_esperadas: list[str]) -> bool:
    """Métrica de fundamentação: a fonte citada casa com a fonte de algum
    chunk esperado?

    Regra: normalização (minúsculas, sem acento/pontuação) + substring em
    qualquer direção. Citações múltiplas separadas por '/', ',', ';' ou
    parênteses também são testadas parte a parte (ex.: 'Banco Central (BCB)').
    """
    citada = str(fonte_citada or "")
    partes = [citada] + re.split(r"[/,;()]", citada)
    partes_norm = [normalizar_fonte(p) for p in partes]
    partes_norm = [p for p in partes_norm if len(p) >= 3]
    if not partes_norm:
        return False

    for fonte_chunk in fontes_esperadas:
        esperada_norm = normalizar_fonte(fonte_chunk)
        if not esperada_norm:
            continue
        for parte in partes_norm:
            if parte in esperada_norm or esperada_norm in parte:
                return True
    return False


def calcular_metricas(detalhes: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrega as métricas do roadmap sobre uma lista de detalhes de execução."""
    total = len(detalhes)
    if total == 0:
        return {
            "n_casos": 0,
            "acertos_veredito": 0,
            "acuracia_veredito_pct": 0.0,
            "taxa_alucinacao_pct": 0.0,
            "conformidade_json_pct": 0.0,
            "latencia_media_seg": 0.0,
            "latencia_fora_dominio_seg": None,
            "n_golpes": 0,
            "fundamentacao_acertos": 0,
            "taxa_fundamentacao_pct": None,
        }

    acertos = sum(d["veredito_obtido"] == d["veredito_esperado"] for d in detalhes)
    alucinacoes = sum(
        d["veredito_esperado"] in ("legitimo", "nao_consta") and d["veredito_obtido"] == "golpe"
        for d in detalhes
    )
    json_ok = sum(bool(d["conformidade_json"]) for d in detalhes)
    latencias = [d["latencia_seg"] for d in detalhes]
    lat_fora = [d["latencia_seg"] for d in detalhes if d["categoria_esperada"] == "fora_de_dominio"]
    golpes = [d for d in detalhes if d["categoria_esperada"] == "golpe"]
    fund_acertos = sum(bool(d["fundamentacao"]) for d in golpes)

    return {
        "n_casos": total,
        "acertos_veredito": acertos,
        "acuracia_veredito_pct": round(acertos / total * 100, 2),
        "taxa_alucinacao_pct": round(alucinacoes / total * 100, 2),
        "conformidade_json_pct": round(json_ok / total * 100, 2),
        "latencia_media_seg": round(sum(latencias) / len(latencias), 3),
        "latencia_fora_dominio_seg": round(sum(lat_fora) / len(lat_fora), 3) if lat_fora else None,
        "n_golpes": len(golpes),
        "fundamentacao_acertos": fund_acertos,
        "taxa_fundamentacao_pct": round(fund_acertos / len(golpes) * 100, 2) if golpes else None,
    }


def executar_benchmark(limite_casos: int | None = None) -> dict[str, Any]:
    """Roda as 3 arquiteturas sobre os dois splits e calcula as métricas do Roadmap.

    ``limite_casos`` (opcional) trunca a lista combinada (principal + stress),
    útil apenas para smoke tests rápidos; a execução oficial usa todos os 35.
    """
    print("=" * 75)
    print("BENCHMARK COMPARATIVO DAS 3 CONFIGURAÇÕES - GOLPE ZERO (35 CASOS)")
    print("=" * 75)

    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        casos_principal = json.load(f)
    with open(TEST_CASES_STRESS_PATH, "r", encoding="utf-8") as f:
        casos_stress = json.load(f)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks_por_id = {c["id"]: c for c in json.load(f)}

    casos: list[dict[str, Any]] = []
    for split, lista in (("principal", casos_principal), ("stress", casos_stress)):
        for tc in lista:
            casos.append({**tc, "_split": split})

    if limite_casos:
        casos = casos[:limite_casos]

    print(f"Split principal: {len(casos_principal)} casos | Split stress: {len(casos_stress)} casos")

    pipeline = PipelineGolpeZero()

    arquiteturas = [
        ("Zero-Shot (Sem Contexto)", pipeline.predizer_zero_shot),
        ("Pipeline RAG Simples", pipeline.predizer_rag_simples),
        ("Agente Decisor (2 Etapas)", pipeline.predizer_agente_decisor),
    ]

    metricas_finais: dict[str, Any] = {}

    for nome_arq, func_predicao in arquiteturas:
        print(f"\n[Avaliando] {nome_arq} sobre {len(casos)} casos...")

        detalhes_execucao: list[dict[str, Any]] = []

        for tc in casos:
            tc_id = tc["id"]
            msg = tc["mensagem_recebida"]
            cat_esperada = tc["categoria_esperada"]
            # Para mensagens fora_de_dominio, o veredito esperado é "nao_consta"
            veredito_esperado = cat_esperada if cat_esperada != "fora_de_dominio" else "nao_consta"

            res = func_predicao(msg)
            veredito_obtido = normalizar_veredito(res.get("veredito", ""))
            fonte_citada = str(res.get("fonte_oficial", "") or "")

            # Fundamentação: somente para casos de golpe com gabarito de chunks
            fundamentacao = None
            if cat_esperada == "golpe":
                fontes_esperadas = [
                    chunks_por_id[cid]["fonte"]
                    for cid in tc.get("chunks_esperados", [])
                    if cid in chunks_por_id
                ]
                fundamentacao = fontes_casam(fonte_citada, fontes_esperadas)

            lat = res.get("_latencia_seg", 0.0)

            detalhes_execucao.append({
                "id": tc_id,
                "split": tc["_split"],
                "categoria_esperada": cat_esperada,
                "veredito_esperado": veredito_esperado,
                "veredito_obtido": veredito_obtido,
                "tipo_golpe_detectado": res.get("tipo_golpe"),
                "fonte_oficial": fonte_citada,
                "fundamentacao": fundamentacao,
                "latencia_seg": lat,
                "conformidade_json": res.get("_conformidade_json", False),
                "score_maximo": res.get("_score_maximo", None),
                "chunks_usados": res.get("_chunks_usados", []),
                "etapa_decisao": res.get("_etapa_decisao", None),
                "proximo_passo": str(res.get("proximo_passo", ""))[:60] + "...",
            })

            marca_fund = "" if fundamentacao is None else (" | FUND:" + ("OK" if fundamentacao else "NAO"))
            print(f"  [{tc_id}|{tc['_split'][:4]}] Esperado: {veredito_esperado:<10} | Obtido: {veredito_obtido:<10} | Latência: {lat:.2f}s{marca_fund}")

        metricas_globais = calcular_metricas(detalhes_execucao)
        metricas_por_split = {
            split: calcular_metricas([d for d in detalhes_execucao if d["split"] == split])
            for split in SPLITS
        }

        metricas_finais[nome_arq] = {
            **metricas_globais,
            "splits": metricas_por_split,
            "detalhes": detalhes_execucao,
        }

    # Exibe a Tabela Comparativa Oficial
    print("\n" + "=" * 110)
    print("PLACAR COMPARATIVO OFICIAL DAS 3 ARQUITETURAS (Critérios 05, 06 e 07)")
    print("=" * 110)
    cabecalho = (
        f"{'Arquitetura':<28} | {'Acc Global':<10} | {'Acc Stress':<10} | {'Alucin.':<8} | "
        f"{'JSON':<7} | {'Fundament.':<10} | {'Lat. Média':<10} | {'Lat. Fora Dom.'}"
    )
    print(cabecalho)
    print("-" * 110)
    for arq, m in metricas_finais.items():
        acc_stress = m["splits"]["stress"]["acuracia_veredito_pct"]
        fund = m["taxa_fundamentacao_pct"]
        lat_fora = m["latencia_fora_dominio_seg"]
        fund_str = f"{fund:.1f}%" if fund is not None else "n/a"
        lat_fora_str = f"{lat_fora:.3f}s" if lat_fora is not None else "n/a"
        print(
            f"{arq:<28} | {m['acuracia_veredito_pct']:>8.1f}% | {acc_stress:>8.1f}% | "
            f"{m['taxa_alucinacao_pct']:>6.1f}% | {m['conformidade_json_pct']:>5.1f}% | "
            f"{fund_str:>10} | {m['latencia_media_seg']:>8.2f}s | {lat_fora_str}"
        )
    print("=" * 110)

    # Persiste em JSON para o relatório (detalhes completos: 35 por config)
    with open(RELATORIO_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metricas_finais, f, ensure_ascii=False, indent=2)
    print(f"\nRelatório de benchmark persistido em: {RELATORIO_OUTPUT}")

    return metricas_finais


if __name__ == "__main__":
    executar_benchmark()