"""Motor de Busca Vetorial Manual e Avaliação de Métricas (Fase 1.2).

Implementa a álgebra linear de similaridade de cosseno pura via NumPy,
avalia Recall@1, Recall@3 sobre o gabarito formal e calibra o limiar empírico.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
BASE_PATH = BASE_DIR / "base" / "chunks_oficiais.json"
TEST_CASES_PATH = BASE_DIR / "avaliacao" / "test_cases.json"
RECALL_OUTPUT_PATH = BASE_DIR / "avaliacao" / "recall_busca.json"
MODEL_NAME = "all-MiniLM-L6-v2"


def similaridade_cosseno(vetor_a: np.ndarray, vetor_b: np.ndarray) -> float:
    """Calcula a similaridade de cosseno manual entre dois vetores:

    Sim(A, B) = (A · B) / (||A|| * ||B||)
    """
    produto_escalar = np.dot(vetor_a, vetor_b)
    norma_a = np.linalg.norm(vetor_a)
    norma_b = np.linalg.norm(vetor_b)

    if norma_a == 0.0 or norma_b == 0.0:
        return 0.0

    return float(produto_escalar / (norma_a * norma_b))


class MotorBuscaVetorial:
    """Motor de recuperação de informação vetorial sem bibliotecas caixa-preta."""

    def __init__(self, caminho_base: Path = BASE_PATH):
        print(f"[Motor] Carregando modelo de embeddings '{MODEL_NAME}'...")
        self.modelo = SentenceTransformer(MODEL_NAME)

        print(f"[Motor] Carregando base de conhecimento de '{caminho_base}'...")
        with open(caminho_base, "r", encoding="utf-8") as f:
            self.chunks: list[dict[str, Any]] = json.load(f)

        print(f"[Motor] Vetorizando {len(self.chunks)} chunks oficiais...")
        textos = [c["conteudo"] for c in self.chunks]
        # Converte para matriz numpy (N, 384)
        self.embeddings_base = np.array(
            self.modelo.encode(textos, show_progress_bar=False)
        )
        print("[Motor] Indexação concluída na memória com sucesso.\n")

    def buscar_top_k(
        self, mensagem: str, k: int = 3
    ) -> list[dict[str, Any]]:
        """Vetoriza a mensagem de consulta e retorna os k chunks com maior similaridade."""
        vetor_msg = self.modelo.encode(mensagem, show_progress_bar=False)

        scores = []
        for i, vetor_chunk in enumerate(self.embeddings_base):
            sim = similaridade_cosseno(vetor_msg, vetor_chunk)
            scores.append((sim, i))

        # Ordena decrescente pelo score de similaridade
        scores.sort(key=lambda x: x[0], reverse=True)

        resultados = []
        for sim, idx in scores[:k]:
            chunk_info = dict(self.chunks[idx])
            chunk_info["score_similaridade"] = round(sim, 4)
            resultados.append(chunk_info)

        return resultados


def avaliar_metricas(
    motor: MotorBuscaVetorial,
    caminho_testes: Path = TEST_CASES_PATH,
    caminho_saida: Path | None = RECALL_OUTPUT_PATH,
) -> dict[str, Any]:
    """Executa a bateria formal de testes para calcular Recall@1, Recall@3

    e calibrar o limiar matemático (Threshold).

    Retorna um dicionário com todas as métricas medidas (recalls, hits por
    caso, estatísticas de score por categoria e limiar sugerido) e, por
    padrão, persiste o resultado em ``avaliacao/recall_busca.json``.
    Passe ``caminho_saida=None`` para apenas calcular sem persistir.
    """
    with open(caminho_testes, "r", encoding="utf-8") as f:
        casos_teste: list[dict[str, Any]] = json.load(f)

    total_golpes = 0
    acertos_top1 = 0
    acertos_top3 = 0

    scores_golpes = []
    scores_legitimos = []
    scores_fora_dominio = []
    detalhes_casos: list[dict[str, Any]] = []

    print("=" * 70)
    print("INICIANDO AUDITORIA EXPERIMENTAL DO MOTOR VETORIAL (RECALL E THRESHOLD)")
    print("=" * 70)

    for tc in casos_teste:
        tc_id = tc["id"]
        msg = tc["mensagem_recebida"]
        cat = tc["categoria_esperada"]
        chunks_esperados = set(tc.get("chunks_esperados", []))

        # Busca Top 3
        top3 = motor.buscar_top_k(msg, k=3)
        top1_chunk_id = top3[0]["id"]
        top3_chunk_ids = [c["id"] for c in top3]
        max_score = top3[0]["score_similaridade"]

        hit_top1 = None
        hit_top3 = None

        if cat == "golpe":
            total_golpes += 1
            scores_golpes.append(max_score)

            hit_top1 = top1_chunk_id in chunks_esperados
            hit_top3 = any(cid in chunks_esperados for cid in top3_chunk_ids)

            if hit_top1:
                acertos_top1 += 1
            if hit_top3:
                acertos_top3 += 1

            status_1 = "HIT" if hit_top1 else "MISS"
            status_3 = "HIT" if hit_top3 else "MISS"

            print(f"[{tc_id}] Golpe: {tc['tipo_golpe_esperado'][:24]:<24} | Score Top 1: {max_score:.4f} | R@1: {status_1} | R@3: {status_3}")

        elif cat == "legitimo":
            scores_legitimos.append(max_score)
            print(f"[{tc_id}] Legítimo: {tc['sinais_criticos'][0][:22]:<24} | Score Máx: {max_score:.4f}")

        else:  # fora_de_dominio
            scores_fora_dominio.append(max_score)
            print(f"[{tc_id}] Fora Domínio: {tc['sinais_criticos'][0][:20]:<24} | Score Máx: {max_score:.4f}")

        detalhes_casos.append({
            "id": tc_id,
            "categoria_esperada": cat,
            "tipo_golpe_esperado": tc.get("tipo_golpe_esperado"),
            "score_top1": max_score,
            "top1_chunk_id": top1_chunk_id,
            "top3_chunk_ids": top3_chunk_ids,
            "chunks_esperados": sorted(chunks_esperados),
            "hit_top1": hit_top1,
            "hit_top3": hit_top3,
        })

    # Cálculo dos Recalls
    recall_1 = (acertos_top1 / total_golpes) * 100 if total_golpes > 0 else 0
    recall_3 = (acertos_top3 / total_golpes) * 100 if total_golpes > 0 else 0

    print("\n" + "=" * 70)
    print("RESULTADOS OFICIAIS DE RECUPERAÇÃO VETORIAL (Critério 03)")
    print("=" * 70)
    print(f"• Total de casos de golpe avaliados : {total_golpes}")
    print(f"• Recall@1 (Top 1 exato)           : {recall_1:.2f}% ({acertos_top1}/{total_golpes})")
    print(f"• Recall@3 (Presente no Top 3)     : {recall_3:.2f}% ({acertos_top3}/{total_golpes})")
    print("-" * 70)

    # Análise Estatística para Calibração do Limiar (Threshold)
    def _estatisticas(scores: list[float]) -> dict[str, float]:
        if not scores:
            return {"media": 0.0, "minimo": 0.0, "maximo": 0.0}
        return {
            "media": round(float(np.mean(scores)), 4),
            "minimo": round(float(np.min(scores)), 4),
            "maximo": round(float(np.max(scores)), 4),
        }

    stats_golpe = _estatisticas(scores_golpes)
    stats_legitimo = _estatisticas(scores_legitimos)
    stats_fora = _estatisticas(scores_fora_dominio)

    media_golpe = stats_golpe["media"]
    min_golpe = stats_golpe["minimo"]
    media_legitimo = stats_legitimo["media"]
    max_legitimo = stats_legitimo["maximo"]
    media_fora = stats_fora["media"]
    max_fora = stats_fora["maximo"]

    print("CALIBRAÇÃO EMPÍRICA DO LIMIAR MATEMÁTICO (THRESHOLD):")
    print(f" - Golpes Reais       -> Média: {media_golpe:.4f} | Mínimo: {min_golpe:.4f} | Máximo: {stats_golpe['maximo']:.4f}")
    print(f" - Casos Legítimos    -> Média: {media_legitimo:.4f} | Mínimo: {stats_legitimo['minimo']:.4f} | Máximo: {max_legitimo:.4f}")
    print(f" - Fora de Domínio    -> Média: {media_fora:.4f} | Mínimo: {stats_fora['minimo']:.4f} | Máximo: {max_fora:.4f}")

    # Limiar ideal sugerido: ponto de corte entre o máximo fora de domínio e o mínimo de golpe
    limiar_ponto_medio = round((max_fora + min_golpe) / 2, 2)
    # Como min_golpe < max_fora, nenhum limiar separa as classes. Adota-se um piso
    # 0.05 abaixo do menor score de golpe: nunca descarta um golpe do gabarito.
    limiar_sugerido = round(min_golpe - 0.05, 2)
    print(f"\n=> PONTO MEDIO (nao separa as classes, classes se sobrepoem): {limiar_ponto_medio}")
    print(f"=> PISO ADOTADO (min golpe {min_golpe:.4f} - 0.05): {limiar_sugerido}")
    print(f"   (Consulta com score < {limiar_sugerido} ativa o veredito compulsório 'não consta')")
    print("=" * 70)

    resultado: dict[str, Any] = {
        "arquivo_testes": str(Path(caminho_testes).resolve().relative_to(BASE_DIR)),
        "modelo_embeddings": MODEL_NAME,
        "total_chunks_base": len(motor.chunks),
        "total_casos": len(casos_teste),
        "total_golpes": total_golpes,
        "acertos_top1": acertos_top1,
        "acertos_top3": acertos_top3,
        "recall_at_1_pct": round(recall_1, 2),
        "recall_at_3_pct": round(recall_3, 2),
        "limiar_sugerido": limiar_sugerido,
        "limiar_ponto_medio": limiar_ponto_medio,
        "scores_por_categoria": {
            "golpe": stats_golpe,
            "legitimo": stats_legitimo,
            "fora_de_dominio": stats_fora,
        },
        "casos": detalhes_casos,
    }

    if caminho_saida is not None:
        caminho_saida = Path(caminho_saida)
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        with open(caminho_saida, "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=2)
        print(f"\n[Artefato] Métricas de recuperação persistidas em: {caminho_saida}")

    return resultado


if __name__ == "__main__":
    motor = MotorBuscaVetorial()
    avaliar_metricas(motor)
