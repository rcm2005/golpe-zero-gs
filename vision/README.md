# Golpe Zero: módulo de Visão Computacional

Disciplina **Applied Computer Vision** (Profs. Arnaldo Viana e Paulo Rodrigues), FIAP Global Solution 2026.2, CitizenTech.

## Integrantes

| Nome completo | RM |
| :--- | :---: |
| Levy Nascimento Junior | 98655 |
| Rafael Autieri dos Anjos | 550885 |
| Allan Von Ivanov | 98705 |
| Luiza Cristina Silva | 99367 |
| Rafael Carvalho Mattos | 99874 |

## O que este módulo faz

Detecta cinco elementos gráficos típicos de golpes em capturas de tela de WhatsApp, SMS, boletos e comprovantes de pessoas 60+: `logo_banco`, `chave_pix`, `link_suspeito`, `qr_code` e `selo_falso`. Faz fine-tuning do YOLOv8n e devolve um JSON com classes, confianças, caixas e nível de risco visual. O produto completo (texto, busca na base oficial e LLM) está no [README principal](../README.md); este módulo alimenta o endpoint `POST /api/analisar/imagem`.

## Entregáveis

| Entregável | Arquivo |
| :--- | :--- |
| Relatório em PDF (título, integrantes, resumo, introdução, objetivo, ferramentas, metodologia, resultados, conclusão) | [RELATORIO_VISAO.pdf](RELATORIO_VISAO.pdf) (fonte em [RELATORIO_VISAO.md](RELATORIO_VISAO.md)) |
| Código completo | [gerador_sintetico.py](gerador_sintetico.py), [train_yolo.py](train_yolo.py), [inference.py](inference.py), [tempo_real.py](tempo_real.py) |
| Dataset anotado em padrão YOLO | [dataset/](dataset/) (120 treino, 30 validação) |
| Pesos e métricas do treino | [weights/best.pt](weights/best.pt), [training_metrics.json](training_metrics.json), [runs/treino_golpe_zero/](runs/treino_golpe_zero/) |
| Vídeo de demonstração | [YouTube (não listado)](https://youtu.be/nZn544MLiM8) · versão anterior em slides: [demo/video_demo_visao.mp4](demo/video_demo_visao.mp4) |

## Resultados (validação, 30 imagens sintéticas)

mAP@50 de 99,5%, mAP@50-95 de 96,24%, precisão de 98,71% e recall de 100%. O dataset é 100% sintético por causa da LGPD, e o relatório registra a limitação: a generalização para prints reais precisa de validação futura. Em CPU, a detecção em vídeo roda a cerca de 14,8 FPS.

## Como reproduzir

Pré-requisitos: Python 3.11+ e `pip install -r requirements.txt` na raiz do repositório. Rode os comandos a partir da raiz.

### Treinar o modelo (YOLOv8n)
```bash
python vision/gerador_sintetico.py
python vision/train_yolo.py
```
*Salva os pesos em `vision/weights/best.pt` e as métricas em `vision/training_metrics.json`.*

### Inferência em imagem e em tempo real
```bash
python vision/inference.py vision/dataset/images/val/golpe_val_0000.jpg
python -m vision.tempo_real --fonte 0 --mostrar          # câmera em tempo real
python -m vision.tempo_real --fonte gravacao.mp4         # vídeo, salva anotado
```
*Salva o print anotado em `vision/output_inferencia/resultado_deteccao.jpg` e imprime o JSON.*

### Reproduzir PDF e vídeo
```bash
python vision/build_pdf.py
python vision/build_video.py   # requer macOS com voz pt-BR (say) para narração
```
