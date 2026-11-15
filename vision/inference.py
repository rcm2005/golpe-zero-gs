"""Módulo de Inferência Visual para Golpe Zero (Applied Computer Vision).

Executa detecção de elementos visuais suspeitos em capturas de tela e comprovantes
usando o modelo YOLOv8n fine-tuned.

As 5 classes monitoradas:
- logo_banco (0): Logotipos de instituições financeiras
- chave_pix (1): Chaves Pix em mensagens de terceiros
- link_suspeito (2): Links encurtados, domínios falsos ou links para download de APK
- qr_code (3): Códigos QR de pagamento ou validação
- selo_falso (4): Selos fraudulentos de autenticidade (Banco Central, Febraban, etc.)

Exporta resumo estruturado em JSON e imagem anotada com bounding boxes.
"""

import json
import os
from pathlib import Path
from typing import Any, Union
import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT_DIR / "vision" / "weights" / "best.pt"
OUTPUT_DIR = ROOT_DIR / "vision" / "output_inferencia"

# Mapeamento de classes e cores (BGR para OpenCV)
CLASS_NAMES = {
    0: "logo_banco",
    1: "chave_pix",
    2: "link_suspeito",
    3: "qr_code",
    4: "selo_falso",
}

CLASS_COLORS = {
    0: (255, 140, 0),    # Azul / Laranja BGR
    1: (46, 125, 50),    # Verde escuro
    2: (0, 0, 220),      # Vermelho
    3: (180, 0, 180),    # Roxo
    4: (0, 140, 255),    # Dourado / Laranja
}


class DetectorVisualGolpe:
    """Motor de inferência de visão computacional com YOLO."""

    def __init__(self, weights_path: Union[str, Path] = DEFAULT_WEIGHTS):
        self.weights_path = Path(weights_path)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.model = self._carregar_modelo()

    def _carregar_modelo(self):
        """Carrega pesos treinados ou modelo base como fallback."""
        if not self.weights_path.exists():
            print(f"[Aviso] Pesos '{self.weights_path}' não encontrados. Usando modelo base 'yolov8n.pt'...")
            return YOLO("yolov8n.pt") if YOLO else None
        print(f"[Visão] Carregando modelo treinado: {self.weights_path}")
        return YOLO(str(self.weights_path)) if YOLO else None

    def analisar_imagem(
        self,
        imagem_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        conf_threshold: float = 0.25,
        salvar_anotada: bool = True,
        nome_saida: str = "resultado_deteccao.jpg"
    ) -> dict[str, Any]:
        """Detecta objetos suspeitos e devolve evidências estruturadas em JSON."""
        # Pré-processa entrada para formato aceito pelo OpenCV/YOLO
        img_cv2 = None
        if isinstance(imagem_input, (str, Path)):
            img_path = Path(imagem_input)
            img_cv2 = cv2.imread(str(img_path))
            if img_cv2 is None:
                raise ValueError(f"Não foi possível abrir a imagem: {img_path}")
        elif isinstance(imagem_input, bytes):
            nparr = np.frombuffer(imagem_input, np.uint8)
            img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(imagem_input, Image.Image):
            img_cv2 = cv2.cvtColor(np.array(imagem_input), cv2.COLOR_RGB2BGR)
        elif isinstance(imagem_input, np.ndarray):
            img_cv2 = imagem_input.copy()

        if img_cv2 is None:
            raise ValueError("Formato de imagem inválido ou não suportado.")

        h_orig, w_orig = img_cv2.shape[:2]

        # Executa inferência
        if self.model is None:
            return {
                "sucesso": False,
                "erro": "Ultralytics YOLO não disponível no ambiente.",
                "elementos_detectados": [],
                "contagem_classes": {},
                "nivel_risco_visual": "DESCONHECIDO",
                "resumo_texto": "Módulo de visão offline.",
            }

        resultados = self.model.predict(img_cv2, conf=conf_threshold, verbose=False)
        resultado = resultados[0]

        elementos_detectados = []
        contagem = {nome: 0 for nome in CLASS_NAMES.values()}
        img_anotada = img_cv2.copy()

        for box in resultado.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            nome_classe = CLASS_NAMES.get(cls_id, f"classe_{cls_id}")

            # Coordenadas em pixels
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            contagem[nome_classe] = contagem.get(nome_classe, 0) + 1
            elementos_detectados.append({
                "classe": nome_classe,
                "classe_id": cls_id,
                "confianca": round(conf, 4),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_normalizado": [
                    round(x1 / w_orig, 4),
                    round(y1 / h_orig, 4),
                    round(x2 / w_orig, 4),
                    round(y2 / h_orig, 4),
                ]
            })

            # Desenha bounding box
            cor = CLASS_COLORS.get(cls_id, (0, 255, 0))
            cv2.rectangle(img_anotada, (x1, y1), (x2, y2), cor, 2)
            rotulo = f"{nome_classe} {conf*100:.1f}%"
            cv2.putText(
                img_anotada, rotulo, (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2, cv2.LINE_AA
            )

        # Salva imagem anotada se solicitado
        caminho_anotada = None
        if salvar_anotada:
            caminho_anotada = OUTPUT_DIR / nome_saida
            cv2.imwrite(str(caminho_anotada), img_anotada)

        # Avaliação heurística do nível de risco visual
        tem_link = contagem.get("link_suspeito", 0) > 0
        tem_selo = contagem.get("selo_falso", 0) > 0
        tem_pix = contagem.get("chave_pix", 0) > 0
        tem_logo = contagem.get("logo_banco", 0) > 0
        tem_qr = contagem.get("qr_code", 0) > 0

        if (tem_link and tem_logo) or (tem_selo and (tem_pix or tem_link)):
            nivel_risco = "CRITICO"
        elif tem_link or tem_selo or (tem_logo and tem_pix):
            nivel_risco = "ALTO"
        elif tem_pix or tem_qr:
            nivel_risco = "MEDIO"
        elif tem_logo:
            nivel_risco = "BAIXO"
        else:
            nivel_risco = "INDETERMINADO"

        # Gera resumo textual para alimentar o pipeline LLM
        itens_resumo = []
        if tem_logo:
            itens_resumo.append(f"Logotipo bancário detectado ({contagem['logo_banco']}x)")
        if tem_link:
            itens_resumo.append(f"Link suspeito ou URL externa não oficial ({contagem['link_suspeito']}x)")
        if tem_selo:
            itens_resumo.append(f"Selo falso de segurança/autenticidade ({contagem['selo_falso']}x)")
        if tem_pix:
            itens_resumo.append(f"Chave Pix exposta na mensagem ({contagem['chave_pix']}x)")
        if tem_qr:
            itens_resumo.append(f"Código QR para pagamento ({contagem['qr_code']}x)")

        resumo_texto = " | ".join(itens_resumo) if itens_resumo else "Nenhum elemento visual de golpe detectado."

        return {
            "sucesso": True,
            "total_elementos": len(elementos_detectados),
            "contagem_classes": contagem,
            "nivel_risco_visual": nivel_risco,
            "resumo_texto": resumo_texto,
            "elementos_detectados": elementos_detectados,
            "caminho_imagem_anotada": str(caminho_anotada) if caminho_anotada else None,
        }


if __name__ == "__main__":
    import sys
    # Se fornecida imagem por linha de comando
    if len(sys.argv) > 1:
        caminho_teste = sys.argv[1]
    else:
        # Usa primeira imagem de validação do dataset
        caminho_teste = str(ROOT_DIR / "vision" / "dataset" / "images" / "val" / "golpe_val_0000.jpg")

    print(f"[Teste de Inferência] Analisando imagem: {caminho_teste}")
    detector = DetectorVisualGolpe()
    resultado_json = detector.analisar_imagem(caminho_teste, salvar_anotada=True)
    print("\n--- RESUMO DA DETECÇÃO VISUAL ---")
    print(json.dumps(resultado_json, indent=2, ensure_ascii=False))
