"""Script de Treinamento e Fine-Tuning do Modelo YOLO (Applied Computer Vision).

Treina o modelo YOLOv8n nas 5 classes de detecção de fraudes financeiras:
0: logo_banco
1: chave_pix
2: link_suspeito
3: qr_code
4: selo_falso

Salva métricas de avaliação (mAP@50, Precision, Recall, Loss) e exporta
os pesos finais para 'vision/weights/best.pt'.
"""

import json
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT_DIR / "vision" / "dataset" / "data.yaml"
WEIGHTS_DIR = ROOT_DIR / "vision" / "weights"
METRICS_PATH = ROOT_DIR / "vision" / "training_metrics.json"


def treinar_modelo_yolo(epochs: int = 10, batch_size: int = 8, imgsz: int = 640):
    """Executa o fine-tuning do YOLOv8n e extrai métricas completas."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("   GOLPE ZERO — TREINAMENTO YOLOv8n (VISÃO)      ")
    print("==================================================")
    print(f"Dataset: {DATA_YAML}")
    print(f"Épocas: {epochs} | Batch: {batch_size} | Resolução: {imgsz}x{imgsz}")

    # Carrega modelo base pré-treinado nano
    model = YOLO("yolov8n.pt")

    # Inicia treinamento
    resultado_treino = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=str(ROOT_DIR / "vision" / "runs"),
        name="treino_golpe_zero",
        exist_ok=True,
        verbose=True,
        workers=2,
        device="cpu",  # Execução portável no macOS
    )

    # Executa validação formal
    print("\n[Validação] Avaliando no split de teste/validação...")
    metricas = model.val()

    # Extrai métricas essenciais
    map50 = float(metricas.box.map50)
    map50_95 = float(metricas.box.map)
    precision = float(metricas.box.mp)
    recall = float(metricas.box.mr)

    print("\n--- RESULTADOS FINAIS DE TREINAMENTO ---")
    print(f"mAP@50:       {map50 * 100:.2f}%")
    print(f"mAP@50-95:    {map50_95 * 100:.2f}%")
    print(f"Precision:    {precision * 100:.2f}%")
    print(f"Recall:       {recall * 100:.2f}%")

    # Copia pesos salvos para vision/weights/best.pt
    run_best_pt = ROOT_DIR / "vision" / "runs" / "treino_golpe_zero" / "weights" / "best.pt"
    destino_best_pt = WEIGHTS_DIR / "best.pt"

    if run_best_pt.exists():
        shutil.copy(run_best_pt, destino_best_pt)
        print(f"[Pesos] Pesos exportados para: {destino_best_pt}")
    else:
        # Fallback de salvamento
        model.save(str(destino_best_pt))
        print(f"[Pesos] Modelo salvo diretamente em: {destino_best_pt}")

    # Salva relatório de métricas em JSON estruturado
    dados_metricas = {
        "modelo": "YOLOv8n-GolpeZero",
        "epocas": epochs,
        "batch_size": batch_size,
        "resolucao_imgsz": imgsz,
        "classes": {
            "0": "logo_banco",
            "1": "chave_pix",
            "2": "link_suspeito",
            "3": "qr_code",
            "4": "selo_falso"
        },
        "map50_pct": round(map50 * 100, 2),
        "map50_95_pct": round(map50_95 * 100, 2),
        "precision_pct": round(precision * 100, 2),
        "recall_pct": round(recall * 100, 2),
        "metricas_por_classe": {}
    }

    # Métricas detalhadas por classe se disponíveis
    if hasattr(metricas.box, "maps") and metricas.box.maps is not None:
        for idx, score in enumerate(metricas.box.maps):
            nome_classe = dados_metricas["classes"].get(str(idx), f"classe_{idx}")
            dados_metricas["metricas_por_classe"][nome_classe] = round(float(score) * 100, 2)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(dados_metricas, f, indent=2, ensure_ascii=False)

    print(f"[Métricas] Métricas salvas com sucesso em: {METRICS_PATH}")
    return dados_metricas


if __name__ == "__main__":
    treinar_modelo_yolo(epochs=10, batch_size=8, imgsz=640)
