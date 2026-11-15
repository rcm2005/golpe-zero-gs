"""Detecção em tempo real do Golpe Zero em vídeo ou câmera.

Lê frames de uma câmera (índice, ex.: 0) ou de um arquivo de vídeo (por exemplo,
uma gravação de tela do WhatsApp), roda o YOLOv8n treinado em cada frame, desenha
as caixas e mede os quadros por segundo.

Uso:
  python vision/tempo_real.py --fonte 0 --mostrar                  # câmera
  python vision/tempo_real.py --fonte gravacao.mp4 --saida out.mp4 # arquivo
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2

from vision.inference import CLASS_NAMES, DEFAULT_WEIGHTS, DetectorVisualGolpe

SAIDA_PADRAO = Path(__file__).resolve().parent / "output_inferencia" / "tempo_real.mp4"


def processar(fonte: str, saida: Path | None, mostrar: bool, conf: float, max_frames: int | None) -> dict:
    cap = cv2.VideoCapture(int(fonte) if fonte.isdigit() else fonte)
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir a fonte de vídeo: {fonte}")
    detector = DetectorVisualGolpe(DEFAULT_WEIGHTS)
    largura, altura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    escritor = None
    if saida:
        saida.parent.mkdir(parents=True, exist_ok=True)
        escritor = cv2.VideoWriter(str(saida), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (largura, altura))

    contagem: Counter = Counter()
    frames, inicio = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and frames >= max_frames):
            break
        res = detector.model.predict(frame, conf=conf, verbose=False)[0]
        for box in res.boxes:
            contagem[CLASS_NAMES[int(box.cls[0])]] += 1
        anotado = res.plot()
        fps = (frames + 1) / (time.time() - inicio)
        cv2.putText(anotado, f"{fps:.1f} FPS", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if escritor:
            escritor.write(anotado)
        if mostrar:
            cv2.imshow("Golpe Zero - tempo real (q para sair)", anotado)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frames += 1
    cap.release()
    if escritor:
        escritor.release()
    cv2.destroyAllWindows()
    total = time.time() - inicio
    resumo = {"frames": frames, "segundos": round(total, 2), "fps_medio": round(frames / total, 2) if total else 0,
              "deteccoes_por_classe": dict(contagem), "saida": str(saida) if saida else None}
    print(resumo)
    return resumo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonte", default="0", help="índice da câmera ou caminho do vídeo")
    ap.add_argument("--saida", type=Path, default=None, help="mp4 anotado de saída")
    ap.add_argument("--mostrar", action="store_true", help="abre janela ao vivo")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    processar(a.fonte, a.saida if a.saida else (None if a.mostrar else SAIDA_PADRAO), a.mostrar, a.conf, a.max_frames)
