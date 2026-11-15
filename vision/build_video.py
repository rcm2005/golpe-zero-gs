#!/usr/bin/env python3
"""Produz vision/demo/video_demo_visao.mp4 (<=180 s, 1280x720, H.264 yuv420p).

Fontes 100% reais:
- Execucoes de `.venv/bin/python vision/inference.py <img val>` (JSON + imagem anotada).
- Metricas de vision/runs/treino_golpe_zero/results.csv e vision/training_metrics.json.
- Contrato real do endpoint POST /api/analisar/imagem (src/api.py).

Pipeline: frames PIL/matplotlib -> narracao pt-BR por blocos (`say -v Luciana`)
-> segmentos de video por duracao + audio mixado -> concat/mux com o ffmpeg
do imageio-ffmpeg. Validacao de duracao/streams via `ffmpeg -i` + decode completo.

Uso:  .venv/bin/python vision/build_video.py
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

import imageio_ffmpeg

VISION = Path(__file__).resolve().parent
ROOT = VISION.parent
DEMO = VISION / "demo"
DEMO.mkdir(parents=True, exist_ok=True)
OUT = DEMO / "video_demo_visao.mp4"
PY = ROOT / ".venv" / "bin" / "python"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H = 1280, 720
FPS = 30
TTS_RATE = 175  # palavras/min, ritmo calmo

# ---------------------------------------------------------------------------
# Fontes / cores
# ---------------------------------------------------------------------------
def _ttf(mono=False, bold=False):
    fam = "DejaVu Sans Mono" if mono else "DejaVu Sans"
    fp = font_manager.FontProperties(family=fam, weight="bold" if bold else "normal")
    return font_manager.findfont(fp, fallback_to_default=False)


def font(size, bold=False, mono=False):
    return ImageFont.truetype(_ttf(mono, bold), size)


BG = (14, 20, 32)
PANEL = (22, 32, 46)
PANEL2 = (18, 26, 38)
TEXT = (232, 238, 245)
MUTED = (159, 176, 195)
ACCENT = (45, 212, 191)
NAVY = (120, 170, 255)
RISK = {"CRITICO": (239, 68, 68), "ALTO": (245, 158, 11),
        "MEDIO": (234, 179, 8), "BAIXO": (34, 197, 94),
        "INDETERMINADO": (148, 163, 184)}
CLASS_RGB = {0: (0, 140, 255), 1: (50, 125, 46), 2: (220, 0, 0),
             3: (180, 0, 180), 4: (255, 140, 0)}
CLASS_NAMES = {0: "logo_banco", 1: "chave_pix", 2: "link_suspeito",
               3: "qr_code", 4: "selo_falso"}


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"cmd falhou: {cmd}\n{r.stderr[-3000:]}")
    return r


def media_duration(path) -> float:
    """Duracao em segundos via parse de `ffmpeg -i` (sem ffprobe)."""
    r = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", r.stderr)
    if not m:
        raise RuntimeError(f"sem Duration para {path}: {r.stderr[-800:]}")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


# ---------------------------------------------------------------------------
# 1) Execucoes REAIS de inferencia
# ---------------------------------------------------------------------------
def run_inferences():
    results = {}
    # 0000 por ultimo: deixa resultado_deteccao.jpg = golpe_val_0000 (usado no PDF)
    for idx in ["0002", "0028", "0000"]:
        img = VISION / "dataset" / "images" / "val" / f"golpe_val_{idx}.jpg"
        log = DEMO / f"inferencia_val_{idx}.log"
        r = run([str(PY), str(VISION / "inference.py"), str(img)])
        log.write_text(r.stdout, encoding="utf-8")
        payload = r.stdout.split("--- RESUMO DA DETECÇÃO VISUAL ---", 1)[1]
        data = json.loads(payload)
        shutil.copy(VISION / "output_inferencia" / "resultado_deteccao.jpg",
                    DEMO / f"anotada_val_{idx}.jpg")
        results[idx] = data
        print(f"[inferencia] {idx}: risco={data['nivel_risco_visual']} "
              f"elementos={data['total_elementos']}")
    return results


# ---------------------------------------------------------------------------
# 2) Metricas reais + charts matplotlib
# ---------------------------------------------------------------------------
def load_metrics():
    rows = list(csv.DictReader((VISION / "runs/treino_golpe_zero/results.csv").open()))
    tm = json.loads((VISION / "training_metrics.json").read_text(encoding="utf-8"))
    return rows, tm


def make_charts(rows, tm):
    plt.rcParams["font.family"] = "DejaVu Sans"
    ep = [int(r["epoch"]) for r in rows]
    series = {
        "Precision": [float(r["metrics/precision(B)"]) * 100 for r in rows],
        "Recall": [float(r["metrics/recall(B)"]) * 100 for r in rows],
        "mAP@50": [float(r["metrics/mAP50(B)"]) * 100 for r in rows],
        "mAP@50-95": [float(r["metrics/mAP50-95(B)"]) * 100 for r in rows],
    }
    fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=130)
    for (name, ys), c in zip(series.items(), ["#2dd4bf", "#7aa2ff", "#f59e0b", "#ef4444"]):
        ax.plot(ep, ys, marker="o", ms=3, lw=2, label=name, color=c)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_xticks(ep)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    ax.set_title("Curvas de validação por época (results.csv)", fontsize=12)
    fig.tight_layout()
    p1 = DEMO / "chart_curvas.png"
    fig.savefig(p1, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=130)
    names = list(tm["metricas_por_classe"].keys())
    vals = [tm["metricas_por_classe"][n] for n in names]
    colors = [tuple(c / 255 for c in CLASS_RGB[i]) for i in range(len(names))]
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    for b, v in zip(bars, vals[::-1]):
        ax.text(b.get_width() + 0.6, b.get_y() + b.get_height() / 2,
                f"{v:.2f}".replace(".", ","), va="center", fontsize=10)
    ax.set_xlim(0, 108)
    ax.set_xlabel("mAP@50-95 (%)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("mAP@50-95 por classe (training_metrics.json) - media = 96,24%",
                 fontsize=12)
    fig.tight_layout()
    p2 = DEMO / "chart_classes.png"
    fig.savefig(p2, facecolor="white")
    plt.close(fig)
    return p1, p2


# ---------------------------------------------------------------------------
# Helpers de frame PIL
# ---------------------------------------------------------------------------
def new_frame():
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def wrap(draw, text, fnt, max_w):
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def text_block(draw, x, y, text, fnt, fill, max_w, lh=None):
    lh = lh or int(fnt.size * 1.35)
    for ln in wrap(draw, text, fnt, max_w):
        draw.text((x, y), ln, font=fnt, fill=fill)
        y += lh
    return y


def title_bar(draw, title, sub=None):
    draw.rectangle([0, 0, W, 92], fill=PANEL)
    draw.rectangle([0, 92, W, 96], fill=ACCENT)
    draw.text((40, 22), title, font=font(30, bold=True), fill=TEXT)
    if sub:
        draw.text((40, 60), sub, font=font(16), fill=MUTED)


def paste_fit(img, path, x, y, w, h, bg=None):
    src = Image.open(path).convert("RGB")
    s = min(w / src.width, h / src.height)
    nw, nh = int(src.width * s), int(src.height * s)
    src = src.resize((nw, nh), Image.LANCZOS)
    if bg:
        draw = ImageDraw.Draw(img)
        draw.rectangle([x, y, x + w, y + h], fill=bg)
    img.paste(src, (x + (w - nw) // 2, y + (h - nh) // 2))


def chip(draw, x, y, color, label, fnt):
    draw.rectangle([x, y + 3, x + 18, y + 21], fill=color)
    draw.text((x + 26, y), label, font=fnt, fill=TEXT)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def frame_abertura():
    img, d = new_frame()
    d.text((60, 70), "GOLPE ZERO", font=font(64, bold=True), fill=ACCENT)
    d.text((60, 150), "Proteção de idosos 60+ contra golpes digitais",
           font=font(30, bold=True), fill=TEXT)
    d.text((60, 196), "FIAP Global Solution 2026.2  |  Applied Computer Vision  |  CitizenTech",
           font=font(17), fill=MUTED)
    d.rectangle([60, 240, 760, 244], fill=ACCENT)
    y = 280
    for b in [
        "Golpistas imitam bancos em prints de WhatsApp e SMS.",
        "Usam logos, chaves Pix, links maliciosos, QR codes e selos falsos.",
        "A visão computacional detecta esses elementos automaticamente.",
        "Módulo YOLOv8n fine-tuned + API REST + Agente Decisor com LLM local.",
    ]:
        d.ellipse([60, y + 8, 72, y + 20], fill=ACCENT)
        y = text_block(d, 90, y, b, font(22), TEXT, 660) + 14
    paste_fit(img, VISION / "dataset/images/val/golpe_val_0002.jpg", 800, 250, 420, 420, bg=PANEL)
    d.text((800, 680), "Exemplo de print sintético (dataset)", font=font(14), fill=MUTED)
    return img


def frame_dataset():
    img, d = new_frame()
    title_bar(d, "Dataset 100% sintético e as 5 classes", "Origem: vision/gerador_sintetico.py (LGPD)")
    y = 130
    for b in [
        "120 imagens de treino / 30 de validação (80/20).",
        "640x640, modos claro e escuro, 4 layouts de golpe.",
        "LGPD: nenhum print real de vítima (dados sensíveis).",
        "Anotação YOLO automática (centroide + dimensões).",
        "Fine-tuning YOLOv8n: 10 épocas, batch 8, CPU.",
    ]:
        d.ellipse([40, y + 8, 52, y + 20], fill=ACCENT)
        y = text_block(d, 70, y, b, font(20), TEXT, 560) + 12
    y += 10
    d.text((40, y), "Classes detectadas:", font=font(20, bold=True), fill=NAVY)
    y += 34
    for i in range(5):
        chip(d, 40, y, CLASS_RGB[i], f"{i}  {CLASS_NAMES[i]}", font(19))
        y += 32
    paste_fit(img, VISION / "dataset/images/val/golpe_val_0002.jpg", 680, 130, 560, 560, bg=PANEL)
    return img


def json_lines(data):
    L = ["{", f'  "sucesso": {str(data["sucesso"]).lower()},',
         f'  "total_elementos": {data["total_elementos"]},',
         '  "contagem_classes": {']
    for k, v in data["contagem_classes"].items():
        L.append(f'    "{k}": {v},')
    L.append("  },")
    L.append(f'  "nivel_risco_visual": "{data["nivel_risco_visual"]}",')
    L.append(f'  "resumo_texto": "{data["resumo_texto"]}",')
    L.append('  "elementos_detectados": [')
    for el in data["elementos_detectados"]:
        L.append("    {")
        L.append(f'      "classe": "{el["classe"]}",')
        L.append(f'      "confianca": {el["confianca"]},')
        L.append(f'      "bbox_xyxy": {el["bbox_xyxy"]},')
        L.append("    },")
    L.append("  ]")
    L.append("}")
    return L


def frame_deteccao(idx, data):
    img, d = new_frame()
    title_bar(d, f"Execução real - golpe_val_{idx}.jpg",
              ".venv/bin/python vision/inference.py vision/dataset/images/val/...")
    paste_fit(img, DEMO / f"anotada_val_{idx}.jpg", 40, 110, 560, 560, bg=PANEL)
    # painel JSON
    px, py, pw, ph = 620, 110, 620, 560
    d.rectangle([px, py, px + pw, py + ph], fill=PANEL2)
    risk = data["nivel_risco_visual"]
    d.rectangle([px + pw - 190, py + 12, px + pw - 12, py + 48], fill=RISK.get(risk, MUTED))
    d.text((px + pw - 180, py + 19), f"RISCO {risk}", font=font(18, bold=True), fill=(255, 255, 255))
    n_el = data['total_elementos']
    d.text((px + 16, py + 16), f"Saída JSON ({n_el} elemento{'s' if n_el != 1 else ''})",
           font=font(17, bold=True), fill=NAVY)
    raw = json_lines(data)
    # escolhe fonte para caber
    size = 14
    while size > 9:
        fnt = font(size, mono=True)
        wrapped = []
        for ln in raw:
            wrapped.extend(wrap(d, ln, fnt, pw - 32) or [""])
        lh = int(size * 1.3)
        if len(wrapped) * lh <= ph - 70:
            break
        size -= 1
    fnt = font(size, mono=True)
    lh = int(size * 1.3)
    y = py + 56
    for ln in wrapped:
        d.text((px + 16, y), ln, font=fnt, fill=(205, 220, 235))
        y += lh
    d.text((40, 682), f"Imagem anotada: vision/output_inferencia/resultado_deteccao.jpg  |  risco {risk}",
           font=font(14), fill=MUTED)
    return img


def frame_curvas(chart, tm):
    img, d = new_frame()
    title_bar(d, "Métricas de treino - results.csv", "Split de validação, por época")
    paste_fit(img, chart, 40, 110, 1200, 430, bg=(255, 255, 255))
    y = 560
    vals = [("mAP@50", f"{tm['map50_pct']}".replace(".", ",")),
            ("mAP@50-95", f"{tm['map50_95_pct']}".replace(".", ",")),
            ("Precision", f"{tm['precision_pct']}".replace(".", ",")),
            ("Recall", f"{tm['recall_pct']}".replace(".", ","))]
    x = 60
    for name, v in vals:
        d.rectangle([x, y, x + 270, y + 110], fill=PANEL)
        d.text((x + 20, y + 16), name, font=font(18), fill=MUTED)
        d.text((x + 20, y + 46), v + "%", font=font(34, bold=True), fill=ACCENT)
        x += 295
    d.text((60, 686), "Valores da época 10 (results.csv), confirmados por training_metrics.json",
           font=font(14), fill=MUTED)
    return img


def frame_classes(chart):
    img, d = new_frame()
    title_bar(d, "mAP@50-95 por classe", "training_metrics.json (metricas_por_classe)")
    paste_fit(img, chart, 40, 110, 1200, 470, bg=(255, 255, 255))
    d.text((60, 600), "A média por classe (96,24%) coincide com o mAP@50-95 geral.",
           font=font(19), fill=TEXT)
    d.text((60, 632), "O mAP@50 geral (99,5%) vem da validação (results.csv / training_metrics.json).",
           font=font(19), fill=TEXT)
    d.text((60, 676), "Precision/recall por classe não são exportados pelos artefatos.",
           font=font(15), fill=MUTED)
    return img


def frame_api():
    img, d = new_frame()
    title_bar(d, "Integração - POST /api/analisar/imagem", "Contrato real de src/api.py (FastAPI)")
    # request
    d.rectangle([40, 120, 600, 250], fill=PANEL)
    d.text((60, 136), "Requisição (multipart/form-data)", font=font(19, bold=True), fill=NAVY)
    d.text((60, 170), "foto: UploadFile  (print / comprovante)", font=font(17, mono=True), fill=TEXT)
    d.text((60, 198), "texto_adicional: str  (opcional)", font=font(17, mono=True), fill=TEXT)
    # fluxo
    d.rectangle([40, 270, 600, 470], fill=PANEL)
    d.text((60, 286), "Fluxo interno", font=font(19, bold=True), fill=NAVY)
    y = 320
    for step in [
        "1. YOLOv8n detecta as 5 classes -> evidências visuais",
        "2. Resumo + nível de risco viram contexto multimodal",
        "3. Agente Decisor (2 etapas, threshold 0,45)",
        "   LLM local Qwen 2.5:3B + cartilhas Febraban / CERT.br",
    ]:
        y = text_block(d, 60, y, step, font(17), TEXT, 520) + 8
    # response
    d.rectangle([640, 120, 1240, 470], fill=PANEL)
    d.text((660, 136), "Resposta (AnaliseResponse)", font=font(19, bold=True), fill=NAVY)
    y = 172
    for fld, desc in [
        ("veredito", "golpe | legitimo | nao_consta"),
        ("tipo_golpe", "classificação do golpe"),
        ("sinais_encontrados", "gatilhos + elementos visuais (classe/confiança)"),
        ("fonte_oficial", "cartilha / norma que fundamenta"),
        ("proximo_passo", "instrução protetiva em linguagem 60+"),
        ("metadados", "modelo_visao, nivel_risco_visual, contagens, latência"),
    ]:
        d.text((660, y), fld, font=font(16, bold=True, mono=True), fill=ACCENT)
        y = text_block(d, 660, y + 24, desc, font(15), MUTED, 560) + 10
    d.rectangle([40, 500, 1240, 640], fill=PANEL2)
    d.text((60, 516), "Exemplo de orientação (proximo_passo):", font=font(18, bold=True), fill=NAVY)
    text_block(d, 60, 550,
               "\"Não clique em links do print e confira no app oficial do banco.\" "
               "Resposta acolhedora e imperativa, voltada ao público 60+.",
               font(19), TEXT, 1160)
    return img


def frame_conclusao(tm):
    img, d = new_frame()
    title_bar(d, "Conclusão", "Golpe Zero - camada visual de proteção ao idoso")
    y = 140
    for b in [
        "Visão computacional leve, executada localmente (CPU), com privacidade.",
        f"mAP@50 {tm['map50_pct']}% e recall {tm['recall_pct']}% na validação (10 épocas).".replace(".", ","),
        "Dataset 100% sintético: respeita a LGPD, sem dados reais de vítimas.",
        "Integrada a API REST + Agente Decisor com LLM local e cartilhas oficiais.",
        "Orientações acolhedoras e protetivas para o público 60+.",
    ]:
        d.ellipse([60, y + 8, 72, y + 20], fill=ACCENT)
        y = text_block(d, 90, y, b, font(23), TEXT, 1100) + 18
    d.rectangle([60, y + 10, 1220, y + 14], fill=ACCENT)
    d.text((60, y + 40), "Código, relatório (PDF) e este vídeo no repositório público.",
           font=font(21), fill=MUTED)
    d.text((60, y + 80), "Muito obrigado!", font=font(40, bold=True), fill=ACCENT)
    return img


# ---------------------------------------------------------------------------
# Narracao pt-BR (TTS macOS, voz Luciana)
# ---------------------------------------------------------------------------
NARRACAO = [
    ("abertura",
     "Olá! Este é o Golpe Zero, projeto da FIAP Global Solution dois mil e vinte e seis, "
     "disciplina Applied Computer Vision. Protegemos pessoas idosas contra golpes digitais: "
     "criminosos imitam bancos em prints de WhatsApp e SMS, com logotipos, chaves Pix e links "
     "maliciosos. Nosso módulo de visão computacional detecta esses elementos automaticamente."),
    ("dataset",
     "O modelo é o YOLOv8 nano, treinado por dez épocas em cinco classes: logotipo de banco, "
     "chave Pix, link suspeito, QR code e selo falso. O dataset é cem por cento sintético: "
     "cento e vinte imagens de treino e trinta de validação, em modos claro e escuro. Usamos "
     "dados sintéticos para respeitar a LGPD, pois prints reais de vítimas conteriam dados "
     "pessoais sensíveis."),
    ("demo0000",
     "Vejam uma execução real do script de inferência sobre uma imagem de validação. O modelo "
     "detectou quatro elementos: logotipo do banco, chave Pix, link suspeito e QR code, todos "
     "com confiança acima de noventa e seis por cento. O resultado é um JSON estruturado com "
     "nível de risco CRÍTICO e a imagem anotada com as caixas delimitadoras."),
    ("demo0002_0028",
     "Nesta segunda imagem, as cinco classes aparecem juntas, incluindo o selo falso, e o "
     "risco também é CRÍTICO. Já nesta terceira, apenas o selo falso é detectado, com risco "
     "ALTO. Cada detecção informa classe, confiança e coordenadas da caixa, em pixels e "
     "normalizadas."),
    ("metricas",
     "As métricas vêm dos artefatos de treino. Na época dez: mAP cinquenta de noventa e nove "
     "vírgula cinco por cento, mAP cinquenta a noventa e cinco de noventa e seis vírgula vinte "
     "e quatro, precisão de noventa e oito vírgula sete e recall de cem por cento. Por classe, "
     "o mAP varia de noventa e dois vírgula nove, na chave Pix, a noventa e nove vírgula zero "
     "oito, no logotipo."),
    ("api",
     "Na integração, a API FastAPI recebe o print no endpoint analisar imagem, como arquivo "
     "multipart, com texto adicional opcional. O YOLO gera evidências visuais que alimentam o "
     "Agente Decisor em duas etapas, com a LLM local Qwen dois ponto cinco, três bilhões, e "
     "cartilhas oficiais da Febraban e do CERT br. A resposta traz veredito, tipo de golpe, "
     "sinais, fonte oficial e o próximo passo, em linguagem acolhedora."),
    ("conclusao",
     "Com visão computacional leve, local e com privacidade, o Golpe Zero oferece uma primeira "
     "camada de proteção contra o estelionato digital. Código, relatório e vídeo no repositório "
     "público. Muito obrigado!"),
]


def synth_narration():
    durations = {}
    for name, text in NARRACAO:
        aiff = DEMO / f"bloco_{name}.aiff"
        run(["say", "-v", "Luciana", "-r", str(TTS_RATE), "-o", str(aiff), text])
        wav = DEMO / f"bloco_{name}.wav"
        run([FFMPEG, "-y", "-i", str(aiff), "-ar", "44100", "-ac", "1",
             "-c:a", "pcm_s16le", str(wav)])
        dur = media_duration(wav)
        durations[name] = dur
        print(f"[tts] {name}: {dur:.2f}s")
    (DEMO / "roteiro_narracao.txt").write_text(
        "\n\n".join(f"[{n}]\n{t}" for n, t in NARRACAO), encoding="utf-8")
    return durations


# ---------------------------------------------------------------------------
# Montagem ffmpeg
# ---------------------------------------------------------------------------
def assemble(frames_durations, audio_durations):
    # frames_durations: lista (path_png, dur)
    seg_list = []
    for i, (png, dur) in enumerate(frames_durations):
        seg = DEMO / f"seg_{i:02d}.mp4"
        run([FFMPEG, "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(png),
             "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast",
             "-tune", "stillimage", "-crf", "20", "-pix_fmt", "yuv420p",
             "-r", str(FPS), "-vf", f"scale={W}:{H}", str(seg)])
        seg_list.append(seg)
    # audio padded por bloco
    pad_list = []
    for name, dur in audio_durations.items():
        wav = DEMO / f"bloco_{name}.wav"
        pad = DEMO / f"pad_{name}.wav"
        run([FFMPEG, "-y", "-i", str(wav), "-af", "apad", "-t", f"{dur:.3f}",
             "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(pad)])
        pad_list.append(pad)
    # concat audio
    alist = DEMO / "audio_concat.txt"
    alist.write_text("\n".join(f"file '{p}'" for p in pad_list), encoding="utf-8")
    full_wav = DEMO / "narracao_full.wav"
    run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(alist),
         "-c", "copy", str(full_wav)])
    # concat video
    vlist = DEMO / "video_concat.txt"
    vlist.write_text("\n".join(f"file '{p}'" for p in seg_list), encoding="utf-8")
    noaudio = DEMO / "video_noaudio.mp4"
    run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(vlist),
         "-c", "copy", str(noaudio)])
    # mux
    run([FFMPEG, "-y", "-i", str(noaudio), "-i", str(full_wav),
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", str(OUT)])
    print(f"[video] {OUT}")


def validate():
    r = subprocess.run([FFMPEG, "-i", str(OUT)], capture_output=True, text=True)
    meta = r.stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", meta)
    assert m, "sem Duration no ffmpeg -i"
    hh, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
    dur = hh * 3600 + mm * 60 + ss
    vstream = re.search(r"Stream #.*Video: (\w+).*?(yuv420p).*?(\d{3,4})x(\d{3,4})", meta)
    astream = re.search(r"Stream #.*Audio: (\w+)", meta)
    print(f"[validacao] duracao={dur:.2f}s")
    print(f"[validacao] video={vstream.group(1) if vstream else '?'} "
          f"{vstream.group(3) if vstream else '?'}x{vstream.group(4) if vstream else '?'} "
          f"{vstream.group(2) if vstream else '?'} | audio={astream.group(1) if astream else '?'}")
    assert dur <= 180.0, f"video excede 180s: {dur}"
    assert vstream and vstream.group(1) == "h264" and vstream.group(2) == "yuv420p"
    assert vstream.group(3) == "1280" and vstream.group(4) == "720"
    assert astream and astream.group(1) == "aac"
    # decode completo
    d = subprocess.run([FFMPEG, "-v", "error", "-i", str(OUT), "-f", "null", "-"],
                       capture_output=True, text=True)
    assert d.returncode == 0 and not d.stderr.strip(), f"decode com erro: {d.stderr[:1000]}"
    print("[validacao] OK: <=180s, 1280x720 h264 yuv420p + aac, decodificavel")
    return dur


def main():
    results = run_inferences()
    rows, tm = load_metrics()
    chart_curvas, chart_classes = make_charts(rows, tm)

    frames = {
        "abertura": frame_abertura(),
        "dataset": frame_dataset(),
        "demo0000": frame_deteccao("0000", results["0000"]),
        "demo0002": frame_deteccao("0002", results["0002"]),
        "demo0028": frame_deteccao("0028", results["0028"]),
        "curvas": frame_curvas(chart_curvas, tm),
        "classes": frame_classes(chart_classes),
        "api": frame_api(),
        "conclusao": frame_conclusao(tm),
    }
    paths = {}
    for name, im in frames.items():
        p = DEMO / f"frame_{name}.png"
        im.save(p)
        paths[name] = p

    audio_dur = synth_narration()
    PAD = 0.3
    block_dur = {n: d + PAD for n, d in audio_dur.items()}

    # mapeamento bloco -> frames (com split)
    mapping = [
        ("abertura", ["abertura"]),
        ("dataset", ["dataset"]),
        ("demo0000", ["demo0000"]),
        ("demo0002_0028", ["demo0002", "demo0028"]),
        ("metricas", ["curvas", "classes"]),
        ("api", ["api"]),
        ("conclusao", ["conclusao"]),
    ]
    frames_durations = []
    for block, fnames in mapping:
        each = block_dur[block] / len(fnames)
        for fn in fnames:
            frames_durations.append((paths[fn], each))
    total = sum(d for _, d in frames_durations)
    print(f"[montagem] duracao total prevista: {total:.2f}s")
    assert total <= 180.0, f"previsto excede 180s: {total}"

    assemble(frames_durations, block_dur)
    dur = validate()
    manifest = {
        "arquivo": str(OUT), "duracao_s": round(dur, 2),
        "resolucao": f"{W}x{H}", "fps": FPS, "codec_video": "h264/yuv420p",
        "codec_audio": "aac", "tts": f"say -v Luciana -r {TTS_RATE}",
        "blocos": {n: round(d, 2) for n, d in audio_dur.items()},
        "inferencias": {k: {"risco": v["nivel_risco_visual"],
                            "total": v["total_elementos"]} for k, v in results.items()},
    }
    (DEMO / "video_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(f"[ok] video_demo_visao.mp4 ({dur:.2f}s)")


if __name__ == "__main__":
    main()