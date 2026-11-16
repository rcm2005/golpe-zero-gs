#!/usr/bin/env python3
"""Gera vision/RELATORIO_VISAO.pdf a partir de vision/RELATORIO_VISAO.md.

Ordem EXATA da rubrica (Applied Computer Vision):
  titulo, integrantes, resumo, introducao, objetivo, explicacao tecnica das
  ferramentas, metodologia passo a passo, exemplos de resultados, conclusao.

- Fonte DejaVuSans (via matplotlib font_manager) para acentos corretos.
- Diagrama do pipeline desenhado com matplotlib (sem mermaid).
- Embed de imagens REAIS dos artefatos de treino/inferencia.
- Metricas conferidas contra results.csv (epoca 10) e training_metrics.json.
- Sem emojis. Validacao final com pypdf (paginas + texto das secoes).

Uso:  .venv/bin/python vision/build_pdf.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from fpdf import FPDF
from fpdf.enums import XPos, YPos

VISION = Path(__file__).resolve().parent
ROOT = VISION.parent
MD_PATH = VISION / "RELATORIO_VISAO.md"
PDF_PATH = VISION / "RELATORIO_VISAO.pdf"
DEMO = VISION / "demo"
DEMO.mkdir(parents=True, exist_ok=True)
DIAGRAM = DEMO / "diagrama_pipeline.png"

# ---------------------------------------------------------------------------
# Fontes DejaVu (matplotlib traz os arquivos TTF)
# ---------------------------------------------------------------------------
def _font(mono: bool = False, weight: str = "normal", style: str = "normal") -> str:
    fam = "DejaVu Sans Mono" if mono else "DejaVu Sans"
    fp = font_manager.FontProperties(family=fam, weight=weight, style=style)
    return font_manager.findfont(fp, fallback_to_default=False)


F_REG = _font()
F_BOLD = _font(weight="bold")
F_OBL = _font(style="italic")
F_BOBL = _font(weight="bold", style="italic")
F_MONO = _font(mono=True)
F_MONOB = _font(mono=True, weight="bold")

# ---------------------------------------------------------------------------
# Cores
# ---------------------------------------------------------------------------
NAVY = (15, 40, 80)
ACCENT = (0, 110, 140)
GREEN = (20, 120, 70)
GRAY = (90, 90, 90)
LIGHT = (244, 245, 247)
CODE_BG = (245, 246, 248)
LINK = (0, 90, 170)


# ---------------------------------------------------------------------------
# Diagrama do pipeline (matplotlib, sem mermaid / sem emojis)
# ---------------------------------------------------------------------------
def build_diagram() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(14, 5.4), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    boxes = [
        # (x, y, w, h, titulo, sub, cor)
        (0.3, 3.3, 3.0, 1.5, "Captura de tela", "WhatsApp / SMS / boleto", "#5b6b7b"),
        (3.7, 3.3, 3.0, 1.5, "OpenCV", "decodificacao BGR -> RGB", "#1f6fb2"),
        (7.1, 3.3, 3.0, 1.5, "YOLOv8n fine-tuned", "PyTorch | 640 x 640", "#0e8a7d"),
        (10.5, 3.3, 3.2, 1.5, "5 classes + confianca", "bounding boxes (bbox)", "#0e8a7d"),
        (10.5, 0.7, 3.2, 1.5, "JSON de risco visual", "nivel + resumo textual", "#7a5ea8"),
        (7.1, 0.7, 3.0, 1.5, "API FastAPI", "POST /api/analisar/imagem", "#7a5ea8"),
        (3.7, 0.7, 3.0, 1.5, "Agente Decisor (2 etapas)", "LLM Qwen 2.5:3B + cartilhas", "#b26a00"),
        (0.3, 0.7, 3.0, 1.5, "Veredito + orientacao", "protetiva ao idoso 60+", "#147846"),
    ]
    centers = []
    for x, y, w, h, title, sub, color in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.6, edgecolor=color, facecolor=color + "22",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color="#10243e")
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=9.5, color="#33475b")
        centers.append((x + w / 2, y + h / 2, x, y, w, h))

    def arrow(p1, p2):
        a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=18,
                            linewidth=1.8, color="#40566b",
                            shrinkA=6, shrinkB=6,
                            connectionstyle="arc3,rad=0.0")
        ax.add_patch(a)

    # linha 1: esquerda -> direita
    for i in range(3):
        _, _, x, y, w, h = centers[i]
        nx = centers[i + 1][2]
        arrow((x + w, y + h / 2), (nx, y + h / 2))
    # descida no fim da linha 1
    _, _, x, y, w, h = centers[3]
    arrow((x + w / 2, y), (x + w / 2, centers[4][3] + centers[4][5]))
    # linha 2: direita -> esquerda
    for i in range(4, 7):
        _, _, x, y, w, h = centers[i]
        nx = centers[i + 1][2] + centers[i + 1][4]
        arrow((x, y + h / 2), (nx, y + h / 2))

    ax.text(7.0, 5.15, "Pipeline de visao computacional - Golpe Zero",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#10243e")
    fig.tight_layout(pad=0.4)
    fig.savefig(DIAGRAM, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[diagrama] {DIAGRAM}")


# ---------------------------------------------------------------------------
# Conferencia das metricas contra os artefatos reais
# ---------------------------------------------------------------------------
def check_metrics(md_text: str) -> dict:
    tm = json.loads((VISION / "training_metrics.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((VISION / "runs/treino_golpe_zero/results.csv").open()))
    e10 = rows[-1]
    assert e10["epoch"] == "10", "ultima epoca do results.csv nao e 10"
    overall = {
        "precision": round(float(e10["metrics/precision(B)"]) * 100, 2),
        "recall": round(float(e10["metrics/recall(B)"]) * 100, 2),
        "map50": round(float(e10["metrics/mAP50(B)"]) * 100, 2),
        "map50_95": round(float(e10["metrics/mAP50-95(B)"]) * 100, 2),
    }
    # batem com o JSON exportado pelo treino
    assert abs(overall["map50"] - tm["map50_pct"]) < 0.05
    assert abs(overall["map50_95"] - tm["map50_95_pct"]) < 0.05
    assert abs(overall["precision"] - tm["precision_pct"]) < 0.05
    assert abs(overall["recall"] - tm["recall_pct"]) < 0.05
    for key, val in overall.items():
        s = f"{val}".replace(".", ",")
        assert s in md_text, f"metrica geral {key}={s} ausente no MD"
    for cls, val in tm["metricas_por_classe"].items():
        s = f"{val}".replace(".", ",")
        assert s in md_text, f"metrica por classe {cls}={s} ausente no MD"
    print("[metricas] conferidas contra results.csv (epoca 10) e training_metrics.json")
    return {"overall": overall, "per_class": tm["metricas_por_classe"]}


# ---------------------------------------------------------------------------
# Parser minimo de Markdown (subconjunto usado no RELATORIO_VISAO.md)
# ---------------------------------------------------------------------------
def parse_md(text: str):
    lines = text.splitlines()
    # cabecalho (titulo + metadados) ate o primeiro '## '
    header = []
    i = 0
    while i < len(lines) and not lines[i].startswith("## "):
        header.append(lines[i])
        i += 1
    sections = []
    cur = None
    for ln in lines[i:]:
        m = re.match(r"^## (\d+)\.\s+(.*)$", ln)
        if m:
            cur = {"num": int(m.group(1)), "title": m.group(2).strip(), "lines": []}
            sections.append(cur)
        elif cur is not None:
            cur["lines"].append(ln)
    return header, sections


INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")


class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 8, f"Golpe Zero - Applied Computer Vision (FIAP GS 2026.2)  |  pag. {self.page_no()}",
                  align="C")

    # ---- texto rico (negrito / codigo / link) ----
    def rich(self, text, size=11, lh=6, indent=0.0, bullet=None, color=(20, 20, 20)):
        base_l = self.l_margin
        self.set_left_margin(base_l + indent)
        self.set_x(base_l + indent)
        if bullet:
            self.set_font("DejaVu", "B", size)
            self.set_text_color(*ACCENT)
            self.write(lh, bullet + "  ")
        for tok in INLINE_RE.split(text):
            if not tok:
                continue
            if tok.startswith("**") and tok.endswith("**"):
                self.set_font("DejaVu", "B", size)
                self.set_text_color(*color)
                self.write(lh, tok[2:-2])
            elif tok.startswith("`") and tok.endswith("`"):
                self.set_font("DejaVuMono", "", size - 0.5)
                self.set_text_color(120, 40, 40)
                self.write(lh, tok[1:-1])
            elif tok.startswith("["):
                m = re.match(r"\[([^\]]+)\]\(([^)]+)\)", tok)
                label = m.group(1) if m else tok
                self.set_font("DejaVu", "", size)
                self.set_text_color(*LINK)
                self.write(lh, label)
            else:
                self.set_font("DejaVu", "", size)
                self.set_text_color(*color)
                self.write(lh, tok)
        self.ln(lh)
        self.set_left_margin(base_l)
        self.set_x(base_l)

    def heading(self, num, title):
        if self.get_y() > self.h - self.b_margin - 30:
            self.add_page()
        self.ln(2)
        self.set_font("DejaVu", "B", 15)
        self.set_text_color(*NAVY)
        self.set_x(self.l_margin)
        self.multi_cell(0, 8, f"{num}. {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.5)
        y = self.get_y() + 1
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(5)

    def subheading(self, title):
        if self.get_y() > self.h - self.b_margin - 20:
            self.add_page()
        self.ln(1)
        self.set_font("DejaVu", "B", 12)
        self.set_text_color(*ACCENT)
        self.set_x(self.l_margin)
        self.multi_cell(0, 6.5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1.5)

    def code_block(self, lines):
        self.set_font("DejaVuMono", "", 8.5)
        self.set_text_color(30, 30, 30)
        pad = 2.0
        lh = 4.4
        avail = self.w - self.l_margin - self.r_margin - 2 * pad
        # altura total
        wrapped = []
        for ln in lines:
            wrapped.extend(self._wrap(ln, avail) or [""])
        total_h = len(wrapped) * lh + 2 * pad
        if self.get_y() + total_h > self.h - self.b_margin:
            self.add_page()
        x0 = self.l_margin
        y0 = self.get_y()
        self.set_fill_color(*CODE_BG)
        self.set_draw_color(220, 222, 225)
        self.rect(x0, y0, self.w - self.l_margin - self.r_margin, total_h, style="DF")
        self.set_xy(x0 + pad, y0 + pad)
        for wl in wrapped:
            self.set_x(x0 + pad)
            self.cell(avail, lh, wl, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_y(y0 + total_h + 3)

    def _wrap(self, text, max_w):
        if not text:
            return [""]
        words = text.split(" ")
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if self.get_string_width(trial) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def image_block(self, path, alt):
        p = (VISION / path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"imagem real ausente: {p}")
        from PIL import Image
        with Image.open(p) as im:
            iw, ih = im.size
        avail_w = self.w - self.l_margin - self.r_margin
        max_h = 120.0
        w = avail_w
        h = w * ih / iw
        if h > max_h:
            h = max_h
            w = h * iw / ih
        if self.get_y() + h + 12 > self.h - self.b_margin:
            self.add_page()
        x = self.l_margin + (avail_w - w) / 2
        y = self.get_y()
        self.image(str(p), x=x, y=y, w=w, h=h)
        self.set_y(y + h + 2)
        self.set_font("DejaVu", "I", 9)
        self.set_text_color(*GRAY)
        self.set_x(self.l_margin)
        self.multi_cell(avail_w, 5, f"Figura: {alt}", align="C",
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    def table_block(self, rows):
        # rows: lista de listas de celulas (header + dados)
        header, data = rows[0], rows[1:]
        ncol = len(header)
        avail = self.w - self.l_margin - self.r_margin
        self.set_font("DejaVu", "", 10)
        # larguras proporcionais ao conteudo
        maxw = []
        for c in range(ncol):
            widths = [self.get_string_width(r[c]) for r in rows if c < len(r)]
            maxw.append(max(widths) + 6)
        s = sum(maxw)
        colw = [max(22.0, avail * mw / s) for mw in maxw]
        # normaliza para avail
        scale = avail / sum(colw)
        colw = [cw * scale for cw in colw]
        pad = 1.5
        lh = 5.0

        def row_height(r, bold=False):
            self.set_font("DejaVu", "B" if bold else "", 10)
            n = 1
            for c in range(ncol):
                txt = r[c] if c < len(r) else ""
                n = max(n, len(self._wrap(txt, colw[c] - 2 * pad)))
            return n * lh + 2 * pad

        def draw_row(r, y, bold=False, fill=None):
            self.set_font("DejaVu", "B" if bold else "", 10)
            rh = row_height(r, bold)
            x = self.l_margin
            for c in range(ncol):
                txt = r[c] if c < len(r) else ""
                if fill:
                    self.set_fill_color(*fill)
                self.set_draw_color(210, 214, 218)
                self.rect(x, y, colw[c], rh, style="D" if not fill else "DF")
                self.set_xy(x + pad, y + pad)
                self.set_text_color(*(NAVY if bold else (25, 25, 25)))
                align = "C" if c > 0 and ncol <= 4 else "L"
                self.multi_cell(colw[c] - 2 * pad, lh, txt, align=align,
                                new_x=XPos.RIGHT, new_y=YPos.TOP)
                x += colw[c]
            return rh

        # quebra de pagina se necessario (cabecalho + 1a linha)
        need = row_height(header, True) + (row_height(data[0]) if data else 0)
        if self.get_y() + need > self.h - self.b_margin:
            self.add_page()
        y = self.get_y()
        y += draw_row(header, y, bold=True, fill=(225, 232, 240))
        for k, r in enumerate(data):
            rh = row_height(r)
            if y + rh > self.h - self.b_margin:
                self.add_page()
                y = self.get_y()
                y += draw_row(header, y, bold=True, fill=(225, 232, 240))
            fill = LIGHT if k % 2 == 0 else None
            y += draw_row(r, y, fill=fill)
        self.set_y(y + 4)


def render_body(pdf: PDF, lines):
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        stripped = ln.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("<!--"):
            # comentario HTML (possivelmente multilinha) - ignorar no PDF
            while i < n and "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        if stripped == "---":
            i += 1
            continue
        if stripped.startswith("### "):
            pdf.subheading(stripped[4:].strip())
            i += 1
            continue
        if stripped.startswith("```"):
            block = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            pdf.code_block(block)
            continue
        mimg = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped)
        if mimg:
            pdf.image_block(mimg.group(2), mimg.group(1))
            i += 1
            continue
        if stripped.startswith("|"):
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                row = lines[i].strip()
                cells = [c.strip() for c in row.strip("|").split("|")]
                # pula linha separadora :---
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    tbl.append(cells)
                i += 1
            if tbl:
                pdf.table_block(tbl)
            continue
        mb = re.match(r"^[-*]\s+(.*)$", stripped)
        if mb:
            pdf.rich(mb.group(1), indent=5, bullet="-")
            i += 1
            continue
        mn = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if mn:
            pdf.rich(mn.group(2), indent=6, bullet=mn.group(1) + ".")
            i += 1
            continue
        # paragrafo: acumula linhas plain consecutivas
        para = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", "|", "```", "![", "- ", "* ", "---", "<!--"))
                    or re.match(r"^\d+\.\s", nxt)):
                break
            para.append(nxt)
            i += 1
        pdf.rich(" ".join(para))
        pdf.ln(1.5)


def build_cover(pdf: PDF, header):
    pdf.add_page()
    title = ""
    meta = []
    for ln in header:
        s = ln.strip()
        if s.startswith("# "):
            title = s[2:].strip()
        elif s and s != "---":
            meta.append(s)
    pdf.ln(30)
    pdf.set_font("DejaVu", "B", 24)
    pdf.set_text_color(*NAVY)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 12, title, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.8)
    y = pdf.get_y()
    pdf.line(pdf.l_margin + 30, y, pdf.w - pdf.r_margin - 30, y)
    pdf.ln(10)
    for k, m in enumerate(meta):
        pdf.set_font("DejaVu", "B" if k == 0 else "", 12 if k == 0 else 11)
        pdf.set_text_color(*(ACCENT if k == 0 else (40, 40, 40)))
        pdf.set_x(pdf.l_margin)
        # remove marcacao ** para o cabecalho
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", m)
        pdf.multi_cell(0, 7, clean, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)


def main():
    md_text = MD_PATH.read_text(encoding="utf-8")
    check_metrics(md_text)
    build_diagram()

    header, sections = parse_md(md_text)
    # Rubrica: titulo, integrantes, resumo, introducao, objetivo, explicacao
    # tecnica, metodologia, exemplos de resultados, conclusao.
    # A secao 9 (Video) e documentacao do repo; o video ja e citado na Conclusao.
    rubric_titles = [
        "Integrantes", "Resumo", "Introdução", "Objetivo",
        "Explicação Técnica das Ferramentas", "Metodologia Passo a Passo",
        "Exemplos de Resultados e Métricas Obtidas", "Conclusão",
    ]
    by_num = {s["num"]: s for s in sections}
    ordered = [by_num[k] for k in range(1, 9)]
    for sec, expect in zip(ordered, rubric_titles):
        assert sec["title"] == expect, f"ordem/titulo inesperado: {sec['title']} != {expect}"

    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)
    pdf.add_font("DejaVu", "", F_REG)
    pdf.add_font("DejaVu", "B", F_BOLD)
    pdf.add_font("DejaVu", "I", F_OBL)
    pdf.add_font("DejaVu", "BI", F_BOBL)
    pdf.add_font("DejaVuMono", "", F_MONO)
    pdf.add_font("DejaVuMono", "B", F_MONOB)
    pdf.set_title("Golpe Zero - Relatorio Tecnico de Visao Computacional Aplicada")
    pdf.set_author("Equipe Golpe Zero - FIAP GS 2026.2")

    build_cover(pdf, header)
    pdf.add_page()
    for sec in ordered:
        pdf.heading(sec["num"], sec["title"])
        render_body(pdf, sec["lines"])

    pdf.output(str(PDF_PATH))
    print(f"[pdf] gerado: {PDF_PATH}")

    # ---- validacao com pypdf ----
    from pypdf import PdfReader
    reader = PdfReader(str(PDF_PATH))
    npages = len(reader.pages)
    text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
    required = [
        "Integrantes", "Resumo", "Introdução", "Objetivo",
        "Explicação Técnica", "Metodologia", "Exemplos de Resultados", "Conclusão",
        "Levy Nascimento Junior", "RM99874", "YOLOv8n", "99,5", "96,24", "98,71", "100,0",
        "99,08", "92,91", "95,56", "98,14", "95,5", "LGPD", "sintético",
    ]
    missing = [r for r in required if r not in text]
    print(f"[pypdf] paginas: {npages}")
    if missing:
        raise SystemExit(f"[pypdf] FALHA - termos ausentes no texto extraido: {missing}")
    print("[pypdf] OK - todas as secoes e metricas presentes no texto extraido")
    assert npages >= 6, "PDF muito curto"
    print(f"[ok] RELATORIO_VISAO.pdf validado ({npages} paginas)")


if __name__ == "__main__":
    main()