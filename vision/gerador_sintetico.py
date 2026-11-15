"""Gerador Sintético de Dataset YOLO para Golpe Zero (Applied Computer Vision).

Gera prints sintéticos hiper-realistas de mensagens de golpes financeiros
(WhatsApp, SMS, boletos falsos, avisos bancários) com anotações automáticas
nas 5 classes exigidas pelo roadmap da FIAP:
0: logo_banco
1: chave_pix
2: link_suspeito
3: qr_code
4: selo_falso
"""

import os
import random
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CLASSES = {
    "logo_banco": 0,
    "chave_pix": 1,
    "link_suspeito": 2,
    "qr_code": 3,
    "selo_falso": 4,
}

BANCOS = [
    ("Banco do Brasil", (255, 230, 0), (0, 45, 114)),     # Amarelo e Azul
    ("Itaú", (236, 112, 0), (0, 51, 153)),                # Laranja e Azul escuro
    ("Bradesco", (204, 9, 47), (255, 255, 255)),          # Vermelho e Branco
    ("Caixa", (0, 92, 169), (243, 148, 35)),              # Azul e Laranja
    ("Nubank", (130, 10, 209), (255, 255, 255)),          # Roxo e Branco
    ("Santander", (236, 0, 0), (255, 255, 255)),          # Vermelho vivo
]

CHAVES_PIX = [
    "11988223344",
    "contato@oficina-rec.com.br",
    "financeiro@central-devolucao.net",
    "049.882.119-00",
    "fbc82991-8821-419b-a019-9941a87b",
    "22998765432",
    "cobrancas.urgentes@sac-seguro.com",
    "pix.titular@banco-pagamentos.net",
]

LINKS_SUSPEITOS = [
    "http://modulo-itau-seguranca.apk",
    "https://bcb-regularizacao-pix.net/liberar",
    "http://bradesco-atualizacao-token.xyz",
    "https://bb-cancelamento-urgente.online",
    "http://suporte-caixa-beneficio.site",
    "https://recadastro-gov-beneficio.com/login",
    "http://seguranca-app-banco.me/download",
    "https://pagamento-multa-desconto.club",
]

SELOS_FALSOS = [
    "SELO OFICIAL BANCO CENTRAL",
    "VERIFICADO FEBRABAN 2026",
    "SEGURANÇA GOV.BR 100% GARANTIDA",
    "HOMOLOGADO SSP-SP",
    "CERTIFICADO DE SEGURANÇA DIGITAL",
]


def desenhar_qr_code(draw: ImageDraw.ImageDraw, x0: int, y0: int, tamanho: int):
    """Desenha padrão geométrico convincente simulando QR Code."""
    # Fundo branco
    draw.rectangle([x0, y0, x0 + tamanho, y0 + tamanho], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    tamanho_celula = max(3, tamanho // 15)
    
    # 3 Marcadores de canto clássicos do QR Code
    def marcador_canto(cx, cy):
        draw.rectangle([cx, cy, cx + tamanho_celula * 4, cy + tamanho_celula * 4], fill=(0, 0, 0))
        draw.rectangle([cx + tamanho_celula, cy + tamanho_celula, cx + tamanho_celula * 3, cy + tamanho_celula * 3], fill=(255, 255, 255))
        draw.rectangle([cx + tamanho_celula * 1.5, cy + tamanho_celula * 1.5, cx + tamanho_celula * 2.5, cy + tamanho_celula * 2.5], fill=(0, 0, 0))

    marcador_canto(x0 + 4, y0 + 4)
    marcador_canto(x0 + tamanho - tamanho_celula * 4 - 4, y0 + 4)
    marcador_canto(x0 + 4, y0 + tamanho - tamanho_celula * 4 - 4)

    # Ruído binário interno
    for i in range(5, 12):
        for j in range(5, 12):
            if random.random() > 0.45:
                x = x0 + i * tamanho_celula
                y = y0 + j * tamanho_celula
                draw.rectangle([x, y, x + tamanho_celula - 1, y + tamanho_celula - 1], fill=(0, 0, 0))


def bbox_to_yolo(box: tuple[int, int, int, int], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Converte (xmin, ymin, xmax, ymax) para formato YOLO normalizado (x_center, y_center, w, h)."""
    xmin, ymin, xmax, ymax = box
    xmin = max(0, min(xmin, img_w - 1))
    ymin = max(0, min(ymin, img_h - 1))
    xmax = max(0, min(xmax, img_w))
    ymax = max(0, min(ymax, img_h))

    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    xc = (xmin + xmax) / (2.0 * img_w)
    yc = (ymin + ymax) / (2.0 * img_h)
    return round(xc, 5), round(yc, 5), round(w, 5), round(h, 5)


def gerar_amostra(idx: int, split: str = "train") -> None:
    """Gera uma imagem sintética única com anotações correspondentes."""
    img_w, img_h = 640, 640
    modo_escuro = random.random() > 0.6
    bg_color = (24, 28, 33) if modo_escuro else (245, 246, 250)
    text_color = (240, 240, 240) if modo_escuro else (25, 25, 25)
    bubble_bg = (38, 48, 60) if modo_escuro else (255, 255, 255)

    img = Image.new("RGB", (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Barra superior de aplicativo (estilo smartphone)
    bar_color = (15, 20, 25) if modo_escuro else (225, 230, 238)
    draw.rectangle([0, 0, img_w, 55], fill=bar_color)
    draw.text((20, 18), "14:32 • 5G • 88%", fill=text_color)
    draw.text((250, 18), "Mensagem Urgente", fill=text_color)

    labels = []  # Lista de tuplas (class_id, xc, yc, w, h)

    # Escolhe tipo de layout de golpe
    tipo_layout = random.choice(["whatsapp", "sms", "notificacao_bancaria", "cobranca_boleto"])

    y_cursor = 75

    # 1. LOGO DE BANCO (Geralmente no topo do card)
    if random.random() > 0.15:
        banco, cor_bg, cor_txt = random.choice(BANCOS)
        logo_x1 = 40
        logo_y1 = y_cursor
        logo_x2 = logo_x1 + random.randint(140, 200)
        logo_y2 = logo_y1 + random.randint(40, 50)

        draw.rectangle([logo_x1, logo_y1, logo_x2, logo_y2], fill=cor_bg, outline=(200, 200, 200), width=1)
        draw.text((logo_x1 + 10, logo_y1 + 12), banco, fill=cor_txt)

        xc, yc, w, h = bbox_to_yolo((logo_x1, logo_y1, logo_x2, logo_y2), img_w, img_h)
        labels.append((CLASSES["logo_banco"], xc, yc, w, h))
        y_cursor = logo_y2 + 25

    # Card / Bolha de mensagem
    card_x1 = 30
    card_y1 = y_cursor
    card_x2 = img_w - 30
    card_y2 = card_y1 + random.randint(320, 420)
    card_y2 = min(card_y2, img_h - 40)

    draw.rectangle([card_x1, card_y1, card_x2, card_y2], fill=bubble_bg, outline=(180, 180, 180), width=1)
    
    # Texto de introdução do golpe
    txt_intro = random.choice([
        "ALERTA: Transação de R$ 3.840,00 suspeita detectada.",
        "Mãe, meu celular quebrou! Salve esse número temporário.",
        "NOTIFICAÇÃO JUDICIAL: Bloqueio imediato da sua chave Pix.",
        "Prezado cliente, sua assinatura expira em 2 horas.",
        "COMUNICADO DE SEGURANÇA: Atualização obrigatória de token.",
    ])
    draw.text((card_x1 + 15, card_y1 + 15), txt_intro, fill=text_color)

    conteudo_y = card_y1 + 55

    # 2. CHAVE PIX (Elemento suspeito comum)
    if random.random() > 0.25:
        chave = random.choice(CHAVES_PIX)
        pix_x1 = card_x1 + 20
        pix_y1 = conteudo_y
        pix_x2 = pix_x1 + 380
        pix_y2 = pix_y1 + 35

        draw.rectangle([pix_x1, pix_y1, pix_x2, pix_y2], fill=(230, 245, 235), outline=(46, 125, 50), width=2)
        draw.text((pix_x1 + 10, pix_y1 + 9), f"PIX: {chave}", fill=(20, 80, 30))

        xc, yc, w, h = bbox_to_yolo((pix_x1, pix_y1, pix_x2, pix_y2), img_w, img_h)
        labels.append((CLASSES["chave_pix"], xc, yc, w, h))
        conteudo_y = pix_y2 + 20

    # 3. LINK SUSPEITO
    if random.random() > 0.20:
        link = random.choice(LINKS_SUSPEITOS)
        link_x1 = card_x1 + 20
        link_y1 = conteudo_y
        link_x2 = link_x1 + min(480, len(link) * 11 + 20)
        link_y2 = link_y1 + 35

        draw.rectangle([link_x1, link_y1, link_x2, link_y2], fill=(235, 240, 255), outline=(25, 118, 210), width=1)
        draw.text((link_x1 + 10, link_y1 + 9), link, fill=(13, 71, 161))

        xc, yc, w, h = bbox_to_yolo((link_x1, link_y1, link_x2, link_y2), img_w, img_h)
        labels.append((CLASSES["link_suspeito"], xc, yc, w, h))
        conteudo_y = link_y2 + 25

    # 4. QR CODE
    if random.random() > 0.35 and conteudo_y + 110 < card_y2:
        qr_tamanho = random.randint(90, 115)
        qr_x1 = card_x1 + 25
        qr_y1 = conteudo_y
        qr_x2 = qr_x1 + qr_tamanho
        qr_y2 = qr_y1 + qr_tamanho

        desenhar_qr_code(draw, qr_x1, qr_y1, qr_tamanho)
        draw.text((qr_x2 + 15, qr_y1 + 30), "Escaneie para pagar\nou validar chave Pix", fill=text_color)

        xc, yc, w, h = bbox_to_yolo((qr_x1, qr_y1, qr_x2, qr_y2), img_w, img_h)
        labels.append((CLASSES["qr_code"], xc, yc, w, h))
        conteudo_y = qr_y2 + 20

    # 5. SELO FALSO (Selo de segurança fraudulento para enganar idosos)
    if random.random() > 0.30 and conteudo_y + 40 < card_y2:
        selo = random.choice(SELOS_FALSOS)
        selo_x1 = card_x1 + 20
        selo_y1 = conteudo_y
        selo_x2 = selo_x1 + 390
        selo_y2 = selo_y1 + 38

        draw.rectangle([selo_x1, selo_y1, selo_x2, selo_y2], fill=(255, 249, 196), outline=(245, 127, 23), width=2)
        draw.text((selo_x1 + 12, selo_y1 + 10), f" {selo} ", fill=(191, 54, 12))

        xc, yc, w, h = bbox_to_yolo((selo_x1, selo_y1, selo_x2, selo_y2), img_w, img_h)
        labels.append((CLASSES["selo_falso"], xc, yc, w, h))

    # Exige pelo menos 1 classe exista na imagem
    if len(labels) == 0:
        # Adiciona um selo falso padrão
        selo_x1, selo_y1, selo_x2, selo_y2 = 50, 150, 450, 200
        draw.rectangle([selo_x1, selo_y1, selo_x2, selo_y2], fill=(255, 249, 196), outline=(245, 127, 23), width=2)
        draw.text((selo_x1 + 15, selo_y1 + 12), " SELO OFICIAL BANCO CENTRAL ", fill=(191, 54, 12))
        xc, yc, w, h = bbox_to_yolo((selo_x1, selo_y1, selo_x2, selo_y2), img_w, img_h)
        labels.append((CLASSES["selo_falso"], xc, yc, w, h))

    # Salva imagem e anotações
    base_dir = Path(__file__).resolve().parent / "dataset"
    img_path = base_dir / "images" / split / f"golpe_{split}_{idx:04d}.jpg"
    lbl_path = base_dir / "labels" / split / f"golpe_{split}_{idx:04d}.txt"

    img.save(img_path, quality=92)
    with open(lbl_path, "w", encoding="utf-8") as f:
        for cid, xc, yc, w, h in labels:
            f.write(f"{cid} {xc:.5f} {yc:.5f} {w:.5f} {h:.5f}\n")


def gerar_todo_dataset(n_train: int = 120, n_val: int = 30):
    """Gera o dataset completo dividido em treino e validação."""
    print(f"[Dataset] Gerando {n_train} imagens de treino e {n_val} de validação...")
    for i in range(n_train):
        gerar_amostra(i, split="train")
    for i in range(n_val):
        gerar_amostra(i, split="val")
    print("[Dataset] Geração concluída com sucesso!")


if __name__ == "__main__":
    gerar_todo_dataset(n_train=120, n_val=30)
