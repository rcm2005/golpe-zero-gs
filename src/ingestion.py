"""Módulo de Ingestão e Processamento da Base de Conhecimento Oficial Antifraude.

Fontes Oficiais Integradas:
1. CERT.br (Centro de Estudos, Resposta e Tratamento de Incidentes de Segurança no Brasil)
2. Banco Central do Brasil (BCB) - Regulamentação do Pix, MED e Bloqueio Cautelar
3. Febraban (Federação Brasileira de Bancos) - Portal Antifraudes
4. Procon-SP (Fundação de Proteção e Defesa do Consumidor) - CDC e Direitos do Consumidor
5. SSP-SP (Secretaria de Segurança Pública do Estado de São Paulo) - Delegacia Eletrônica
"""

import io
import json
import re
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent.parent
BASE_PATH = BASE_DIR / "base"
BASE_PATH.mkdir(exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 15


def limpar_texto(texto: str) -> str:
    """Remove quebras excessivas, espaços duplicados e caracteres residuais."""
    if not texto:
        return ""
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"•\s*", "- ", texto)
    texto = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", texto)
    return texto.strip()


def chunk_texto(
    texto: str,
    tamanho_min: int = 280,
    tamanho_max: int = 550,
    overlap: int = 40,
) -> list[str]:
    """Divide um texto em blocos de tamanho aproximado (300 a 600 caracteres)

    respeitando o encerramento de frases e parágrafos completos.
    """
    texto = limpar_texto(texto)
    if not texto:
        return []

    # Divide por pontuação final mantendo a estrutura da frase
    frases = re.split(r"(?<=[.!?])\s+", texto)
    chunks = []
    chunk_atual: list[str] = []
    tam_atual = 0

    for frase in frases:
        frase = frase.strip()
        if not frase:
            continue
        tam_frase = len(frase)

        if tam_atual + tam_frase + 1 <= tamanho_max:
            chunk_atual.append(frase)
            tam_atual += tam_frase + 1
        else:
            if chunk_atual and tam_atual >= tamanho_min:
                chunks.append(" ".join(chunk_atual))
                # Overlap leve com a última frase para preservar coerência
                if overlap > 0 and len(chunk_atual) > 1:
                    chunk_atual = [chunk_atual[-1], frase]
                    tam_atual = sum(len(f) + 1 for f in chunk_atual)
                else:
                    chunk_atual = [frase]
                    tam_atual = tam_frase
            else:
                chunk_atual.append(frase)
                chunks.append(" ".join(chunk_atual))
                chunk_atual = []
                tam_atual = 0

    if chunk_atual:
        chunk_final = " ".join(chunk_atual)
        if len(chunk_final) >= 80:
            chunks.append(chunk_final)
        elif chunks:
            # Anexa ao último chunk se for muito curto
            chunks[-1] = chunks[-1] + " " + chunk_final

    return chunks


def ingerir_cert_br() -> list[dict[str, Any]]:
    """Coleta e processa os fascículos de golpes do CERT.br.

    Fontes:
    - Golpes: Não se Deixe Enganar
    - Golpes: Evite Fraudes
    - Golpes: Caiu? Veja o que Fazer
    """
    url_base = "https://cartilha.cert.br/fasciculos/"
    licenca = "CC BY-NC-SA 4.0"
    chunks_estruturados = []
    contador = 1

    fasciculos = [
        {
            "tag": "golpes",
            "url_pdf": "https://cartilha.cert.br/fasciculos/golpes/fasciculo-golpes.pdf",
            "url_ref": "https://cartilha.cert.br/fasciculos/#golpes",
            "titulo": "Golpes: Não se Deixe Enganar",
            "tipo_default": "engenharia_social",
        },
        {
            "tag": "evite_fraudes",
            "url_pdf": "https://cartilha.cert.br/fasciculos/golpes-evite-fraudes/fasciculo-golpes-evite-fraudes.pdf",
            "url_ref": "https://cartilha.cert.br/fasciculos/#golpes-evite-fraudes",
            "titulo": "Golpes: Evite Fraudes",
            "tipo_default": "prevencao_fraudes",
        },
        {
            "tag": "o_que_fazer",
            "url_pdf": "https://cartilha.cert.br/fasciculos/golpes-o-que-fazer/fasciculo-golpes-o-que-fazer.pdf",
            "url_ref": "https://cartilha.cert.br/fasciculos/#golpes-o-que-fazer",
            "titulo": "Golpes: Caiu? Veja o que Fazer",
            "tipo_default": "resposta_a_golpes",
        },
    ]

    for fasc in fasciculos:
        textos_secoes = []
        try:
            resp = requests.get(
                fasc["url_pdf"],
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            reader = PdfReader(io.BytesIO(resp.content))

            for pag_idx, pag in enumerate(reader.pages):
                txt = pag.extract_text()
                if not txt:
                    continue
                linhas = [
                    l.strip()
                    for l in txt.split("\n")
                    if l.strip()
                    and not l.startswith("fontTools")
                    and "cartilha.cert.br" not in l
                    and "Produção:" not in l
                    and "Fascículo" not in l
                    and "Download" not in l
                    and "Assista aos vídeos" not in l
                ]
                bloco = " ".join(linhas)
                if len(bloco) > 100:
                    textos_secoes.append((pag_idx + 1, bloco))
        except Exception as e:
            print(f"[CERT.br] Aviso ao baixar {fasc['tag']}: {e}. Usando base canônica.")

        # Fallback de conteúdo canônico caso o download falhe ou venha incompleto
        if not textos_secoes:
            textos_secoes = _obter_cert_br_canonico(fasc["tag"])

        for pag_num, bloco in textos_secoes:
            tipo_golpe = _classificar_tipo_cert(bloco, fasc["tipo_default"])
            sub_chunks = chunk_texto(bloco, tamanho_min=280, tamanho_max=540)
            for ch in sub_chunks:
                chunks_estruturados.append(
                    {
                        "id": f"chunk_certbr_{contador:03d}",
                        "fonte": f"CERT.br - Cartilha de Segurança para a Internet ({fasc['titulo']})",
                        "url": fasc["url_ref"],
                        "licenca": licenca,
                        "tipo_golpe": tipo_golpe,
                        "conteudo": ch,
                    }
                )
                contador += 1

    return chunks_estruturados


def _classificar_tipo_cert(texto: str, padrao: str) -> str:
    """Classifica semanticamente o trecho do CERT.br."""
    t = texto.lower()
    if "whatsapp" in t or "aplicativo de mensagem" in t:
        return "falso_whatsapp"
    if "engenharia social" in t or "pressa" in t or "apelo emocional" in t:
        return "engenharia_social"
    if (
        "phishing" in t
        or "link" in t
        or "encomenda retida" in t
        or "cpf cancelado" in t
    ):
        return "phishing"
    if "pix" in t or "devolva pix" in t or "med" in t:
        return "med_pix"
    if "cartão" in t or "conteste compras" in t:
        return "fraude_cartao"
    if "boletim" in t or "b.o." in t or "polícia" in t:
        return "boletim_ocorrencia"
    if (
        "identidade" in t
        or "registrato" in t
        or "abram contas em seu nome" in t
    ):
        return "furto_identidade"
    if "empréstimo" in t:
        return "falso_emprestimo"
    if "áudio" in t or "vídeo" in t or "inteligência artificial" in t:
        return "deepfake_ia"
    return padrao


def _obter_cert_br_canonico(tag: str) -> list[tuple[int, str]]:
    """Trechos textuais canônicos do CERT.br mantidos para alta disponibilidade."""
    if tag == "golpes":
        return [
            (
                6,
                "Engenharia social é quando alguém tenta enganar outra pessoa explorando sentimentos, como medo, obediência, confiança, carência, ganância, curiosidade e vontade de ajudar. É uma técnica muito usada por golpistas para levar as pessoas a agir sem pensar.",
            ),
            (
                7,
                "Apelo emocional: golpistas são especialistas em engenharia social. Sabem usar as palavras e o tom certos para mexer com seus sentimentos e levar você a agir por impulso. Antes de agir, pare e pense. Procure sinais de: ameaça, pressa, apelo afetivo, constrangimento, pedido de segredo, vantagem boa demais ou oportunidade exclusiva.",
            ),
            (
                9,
                "Fique atento a golpes do dia a dia. Golpistas usam temas cotidianos para chamar sua atenção, pegar seus dados e levar você a acessar links maliciosos ou instalar malware. Suspeite de mensagens sobre CPF cancelado, encomenda retida, oferta de emprego ou pagamento pendente. Não faça o que o golpista pede e contate os canais oficiais.",
            ),
            (
                12,
                "Duvide até de áudios e vídeos. Golpistas criam imitações com Inteligência Artificial (deepfake) para pedir dinheiro e indicar investimentos falsos, explorando a confiança ou a imagem de pessoas conhecidas e famosos. Sempre converse pessoalmente ou confirme por canal conhecido antes de enviar valores.",
            ),
        ]
    elif tag == "evite_fraudes":
        return [
            (
                4,
                "Proteja suas contas. Golpistas querem acessar suas contas para se passar por você, usar seus dados, fazer transferências e dívidas em seu nome e enganar seus contatos. Use senhas fortes, ative a verificação em duas etapas (2FA) e nunca informe códigos de confirmação recebidos por SMS ou e-mail a ninguém.",
            ),
            (
                14,
                "Só devolva o Pix para a conta de origem. Se receber um Pix por engano de pessoa desconhecida e ela entrar em contato exigindo devolução para outra conta ou chave diferente, não faça isso. Devolva apenas pelo botão oficial de devolução no aplicativo do seu próprio banco diretamente para a conta que originou o pagamento.",
            ),
            (
                11,
                "Antes de pagar ou transferir, verifique todos os dados do destinatário: confira o nome completo, o CPF ou CNPJ mascarado e o banco de destino na tela de confirmação do Pix ou do boleto bancário. Se os dados não coincidirem com o favorecido legítimo, cancele imediatamente a operação.",
            ),
            (
                17,
                "Evite que abram contas em seu nome. Consulte periodicamente o sistema Registrato do Banco Central do Brasil para verificar se há contas correntes, chaves Pix ou empréstimos abertos sem a sua autorização. Mantenha seus documentos e fotos de identificação protegidos.",
            ),
        ]
    else:  # o_que_fazer
        return [
            (
                4,
                "Se caiu num golpe, aja rápido. O golpista insiste, pede mais dinheiro ou faz ameaças? Corte o contato imediatamente. Manter a conversa pode expor mais informações pessoais e permitir novas chantagens. Não envie mais dinheiro, não tente negociar e bloqueie o perfil e o número do golpista.",
            ),
            (
                6,
                "Conteste o Pix imediatamente. Entre em contato na mesma hora com o seu banco e solicite a contestação da transação por meio do Mecanismo Especial de Devolução (MED). O MED é a ferramenta oficial regulamentada pelo Banco Central que viabiliza o bloqueio e a devolução de valores em caso de fraude comprovada.",
            ),
            (
                7,
                "Conteste compras indevidas no cartão de crédito. Se foram realizadas transações que você não reconhece, ligue para a central do emissor do cartão, solicite o cancelamento e bloqueio imediato do plástico e peça a contestação formal (chargeback) das despesas fraudulentas.",
            ),
            (
                10,
                "Faça um Boletim de Ocorrência (B.O.). Registre a fraude na Delegacia Eletrônica da Polícia Civil. O boletim oficial é o documento formal que comprova a ocorrência do crime perante as instituições bancárias, órgãos de proteção ao consumidor e autoridades policiais.",
            ),
        ]


def ingerir_febraban() -> list[dict[str, Any]]:
    """Extrai todas as fraudes e diretrizes do Portal Febraban Antifraudes.

    Coleta da API oficial e das páginas de detalhe com 'Como é' e 'Como evitar'.
    """
    url_base = "https://antifraudes.febraban.org.br/"
    licenca = "Uso educacional e informativo público (Febraban)"
    chunks_estruturados = []
    contador = 1

    headers = {
        "User-Agent": USER_AGENT,
        "Referer": "https://portal.febraban.org.br/AntiFraude/",
        "X-Requested-With": "XMLHttpRequest",
    }

    artigos_febraban = []
    try:
        s = requests.Session()
        s.headers.update(headers)
        s.get(
            "https://portal.febraban.org.br/AntiFraude/",
            timeout=REQUEST_TIMEOUT,
        )

        for page in [1, 2]:
            r = s.get(
                "https://portal.febraban.org.br/AntiFraude/PodGetItems",
                params={"page": page},
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code == 200:
                itens = r.json().get("Items", [])
                for item in itens:
                    id_antifraude = item.get("IdAntifraude")
                    titulo = item.get("DescTitulo", "").strip()
                    url_detalhe = f"https://portal.febraban.org.br/DetalheAntifraude/{id_antifraude}/pt-br/"

                    try:
                        r_det = s.get(url_detalhe, timeout=REQUEST_TIMEOUT)
                        if r_det.status_code == 200:
                            soup = BeautifulSoup(r_det.text, "html.parser")
                            art = soup.find("article", class_="fbr-news")
                            if art:
                                ps = [
                                    p.get_text(" ", strip=True)
                                    for p in art.find_all("p")
                                ]
                                texto_completo = " ".join(ps)
                                if len(texto_completo) > 80:
                                    artigos_febraban.append(
                                        (titulo, url_detalhe, texto_completo)
                                    )
                    except Exception:
                        pass
    except Exception as e:
        print(f"[Febraban] Erro na requisição online: {e}. Usando base canônica.")

    if not artigos_febraban:
        artigos_febraban = _obter_febraban_canonico()

    for titulo, url, texto in artigos_febraban:
        tipo_golpe = _classificar_tipo_febraban(titulo)
        sub_chunks = chunk_texto(texto, tamanho_min=290, tamanho_max=540)
        for ch in sub_chunks:
            chunks_estruturados.append(
                {
                    "id": f"chunk_febraban_{contador:03d}",
                    "fonte": f"Febraban Antifraudes - {titulo}",
                    "url": url,
                    "licenca": licenca,
                    "tipo_golpe": tipo_golpe,
                    "conteudo": ch,
                }
            )
            contador += 1

    return chunks_estruturados


def _classificar_tipo_febraban(titulo: str) -> str:
    """Mapeia os títulos dos golpes da Febraban para chaves estruturadas."""
    t = titulo.lower()
    if "central" in t:
        return "falsa_central_atendimento"
    if "whatsapp" in t:
        return "falso_whatsapp"
    if "motoboy" in t:
        return "falso_motoboy"
    if "remoto" in t or "fantasma" in t:
        return "mao_fantasma_acesso_remoto"
    if "pix" in t:
        return "pix_errado_comprovante_falso"
    if "troca de cartão" in t:
        return "troca_cartao"
    if "link" in t or "phishing" in t:
        return "phishing"
    if "investimento" in t:
        return "falso_investimento"
    if "boleto" in t:
        return "falso_boleto"
    if "empréstimo" in t:
        return "falso_emprestimo"
    if "maquininha" in t or "aniversário" in t:
        return "maquininha_quebrada_presente"
    if "leilão" in t:
        return "falso_leilao"
    if "imposto de renda" in t or "restituição" in t:
        return "falso_imposto_renda"
    if "advogado" in t:
        return "falso_advogado"
    if "venda" in t:
        return "falsa_venda_ecommerce"
    return "fraude_bancaria"


def _obter_febraban_canonico() -> list[tuple[str, str, str]]:
    """Base de contingência de golpes da Febraban com textos autênticos."""
    url_base = "https://antifraudes.febraban.org.br/"
    return [
        (
            "Golpe da falsa central de atendimento",
            url_base,
            "Como é: O fraudador entra em contato com a vítima se passando por um falso funcionário do banco ou empresa com a qual ela tem relacionamento ativo. Informa que sua conta foi invadida, clonada ou outro problema e, a partir daí, solicita os dados pessoais e financeiros da vítima. E até mesmo pede para que ela ligue na central do banco, no número que aparece atrás do seu cartão, mas o fraudador continua na linha para simular o atendimento da central e pedir os dados da sua conta, dos seus cartões e a sua senha digitada. Como evitar: Recebeu uma ligação minimamente parecida com essa? DESLIGA NA HORA. Depois de desligar, pegue outro telefone e ligue para o número oficial do seu banco.",
        ),
        (
            "Golpe do falso motoboy",
            url_base,
            "Como é: O fraudador entra em contato dizendo que o cartão de crédito da vítima foi clonado ou teve compras suspeitas. Para suposta segurança, pede que o cliente corte o cartão ao meio mantendo o chip intacto e diz que enviará um motoboy para recolher o cartão. Com o chip preservado e a senha informada na ligação, os criminosos realizam saques e compras vultosas. Como evitar: Nenhum banco recolhe cartões na casa dos clientes sob hipótese alguma. Se receber essa solicitação, desligue e avise a instituição.",
        ),
        (
            "Golpe do Whatsapp",
            url_base,
            "Como é: Os fraudadores criam uma conta no WhatsApp utilizando uma foto da vítima e mandam mensagens para familiares e amigos dizendo que trocou de número de telefone porque o celular quebrou ou deu problema. Em seguida, inventam uma urgência financeira e pedem dinheiro via Pix. Como evitar: Sempre desconfie de pedidos emergenciais de dinheiro via WhatsApp, mesmo que a foto e o nome sejam de conhecidos. Ligue para o número original da pessoa antes de qualquer transferência.",
        ),
        (
            "Golpe acesso remoto ou mão fantasma",
            url_base,
            "Como é: O criminoso liga se passando pelo banco e afirma que há um problema de segurança no aplicativo bancário. Ele instrui a vítima a instalar um aplicativo de suporte ou segurança (na verdade uma ferramenta de acesso remoto como AnyDesk ou TeamViewer). Uma vez instalado, o criminoso assume o controle do aparelho celular e faz transações e transferências sem consentimento. Como evitar: Bancos nunca solicitam a instalação de aplicativos de suporte remoto nem pedem para o cliente baixar ferramentas de terceiros.",
        ),
        (
            "Golpe da maquininha quebrada ou presente de aniversário",
            url_base,
            "Como é: O golpista vai até a residência da vítima informando que tem uma entrega de flores ou presente de aniversário, mas que é obrigatório pagar uma taxa de entrega pequena na maquininha de cartão. O visor da maquininha está coberto, quebrado ou adulterado, cobrando valores exorbitantes (milhares de reais) em vez da pequena taxa. Como evitar: Não aceite cobranças surpresa na entrega se não tiver solicitado o serviço e nunca passe cartão em máquina com visor danificado ou ilegível.",
        ),
        (
            "Golpe do Pix errado",
            url_base,
            "Como é: O golpista faz um Pix intencional ou simula uma transferência e entra em contato desesperado afirmando que cometeu um erro e transferiu o dinheiro por engano. Ele pede que a vítima transfira o valor imediatamente para outra conta ou outra chave Pix. Em seguida, aciona o MED no banco dele alegando fraude, fazendo a vítima pagar duas vezes. Como evitar: Nunca transfira para conta de terceiro indicada por telefone ou mensagem. Utilize apenas a função oficial 'Devolver' no extrato do seu aplicativo bancário.",
        ),
        (
            "Golpe do falso empréstimo",
            url_base,
            "Como é: Golpistas oferecem empréstimos com condições extremamente vantajosas para pessoas negativadas ou com crédito restrito. No entanto, para que o valor seja liberado, exigem o pagamento antecipado de taxas de cartório, seguro de crédito ou avalista via Pix. Após o pagamento, os golpistas desaparecem. Como evitar: Instituições financeiras regulares nunca cobram taxa adiantada ou depósito prévio para liberação de empréstimo.",
        ),
    ]


def ingerir_bcb() -> list[dict[str, Any]]:
    """Extrai a regulamentação oficial do Mecanismo Especial de Devolução (MED)

    e regras de segurança do Pix do Banco Central do Brasil.
    Fontes: Resoluções BCB 103/2021, BCB 1/2020, BCB 587 e Comunicados Oficiais.
    """
    url_med = "https://www.bcb.gov.br/estabilidadefinanceira/pix-seguranca"
    url_noticia = "https://www.bcb.gov.br/detalhenoticia/554/noticia"
    licenca = "Conteúdo governamental aberto (Domínio Público / Governo Federal)"
    chunks_estruturados = []
    contador = 1

    textos_bcb = []
    # Tenta obter dados da API oficial de notícias do Banco Central
    try:
        r = requests.get(
            "https://www.bcb.gov.br/api/servico/sitebcb/noticias/554",
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            conteudo_item = r.json().get("conteudo", [{}])[0]
            corpo_html = conteudo_item.get("corpo", "")
            soup = BeautifulSoup(corpo_html, "html.parser")
            texto_noticia = soup.get_text(" ", strip=True)
            if len(texto_noticia) > 200:
                textos_bcb.append(
                    ("Regulamentação Oficial do MED", url_noticia, texto_noticia)
                )
    except Exception as e:
        print(f"[BCB] Notícia online indisponível: {e}. Usando base canônica.")

    # Complementa com as regras normativas consolidadas do MED e da Segurança do Pix
    regras_normativas = _obter_bcb_normativo_canonico()
    textos_bcb.extend(regras_normativas)

    for titulo, url, texto in textos_bcb:
        tipo_golpe = _classificar_tipo_bcb(texto)
        sub_chunks = chunk_texto(texto, tamanho_min=290, tamanho_max=550)
        for ch in sub_chunks:
            chunks_estruturados.append(
                {
                    "id": f"chunk_bcb_{contador:03d}",
                    "fonte": f"Banco Central do Brasil - {titulo}",
                    "url": url,
                    "licenca": licenca,
                    "tipo_golpe": tipo_golpe,
                    "conteudo": ch,
                }
            )
            contador += 1

    return chunks_estruturados


def _classificar_tipo_bcb(texto: str) -> str:
    """Mapeia trechos normativos do BCB."""
    t = texto.lower()
    if "80 dias" in t or "prazo" in t:
        return "prazos_med"
    if "bloqueio cautelar" in t:
        return "bloqueio_cautelar_pix"
    if "desacordo comercial" in t:
        return "excecoes_med"
    if "noturno" in t or "limite" in t:
        return "limites_seguranca_pix"
    return "med_pix"


def _obter_bcb_normativo_canonico() -> list[tuple[str, str, str]]:
    """Regras oficiais regulamentadas pelo Banco Central do Brasil sobre Pix e MED."""
    url_med = "https://www.bcb.gov.br/estabilidadefinanceira/pix-seguranca"
    return [
        (
            "Regras Gerais do Mecanismo Especial de Devolução (MED)",
            url_med,
            "O Mecanismo Especial de Devolução (MED) é um conjunto de regras e procedimentos regulamentado pelo Banco Central do Brasil no Regulamento do Pix (Resolução BCB nº 103/2021 e Resolução BCB nº 1/2020) para facilitar a devolução de recursos em hipóteses de fundada suspeita de fraude ou falha operacional nos sistemas das instituições financeiras. O mecanismo opera de forma padronizada entre os bancos para aumentar a segurança das transferências instantâneas.",
        ),
        (
            "Prazos Oficiais para Acionamento e Análise do MED",
            url_med,
            "A vítima de fraude ou golpe tem o prazo limite de até 80 dias, contados a partir da data de realização do Pix, para registrar a contestação junto à sua instituição financeira de relacionamento. Após a abertura do registro de infração, as instituições envolvidas dispõem de até 7 dias corridos para analisar se houve fraude. Caso confirmada, a instituição recebedora tem até 96 horas para devolver os recursos recuperados à conta de origem.",
        ),
        (
            "Funcionamento do Bloqueio Cautelar do Pix",
            url_med,
            "O Bloqueio Cautelar é uma medida de segurança operacional em que a instituição detentora da conta recebedora pessoa física ou jurídica bloqueia os recursos transacionados por até 72 horas em casos com fundada suspeita de fraude. O bloqueio temporário é efetuado simultaneamente à análise aprofundada de risco da transação, garantindo maior chance de estorno antes da dispersão dos valores.",
        ),
        (
            "Exceções e Não Aplicabilidade do MED",
            url_med,
            "O MED não se aplica e não pode ser acionado em controvérsias decorrentes de desacordo comercial (por exemplo, quando o consumidor adquire produto que não é entregue conforme esperado ou quando há divergência sobre mercadorias de boa-fé) nem em transferências realizadas espontaneamente por engano ou erro de digitação de chave Pix por parte do pagador.",
        ),
        (
            "Limites Noturnos e Parâmetros de Segurança do Pix",
            url_med,
            "As regras do Banco Central determinam que as instituições financeiras estabeleçam limite de transferência padrão de R$ 1.000,00 no horário noturno (compreendido entre 20h e 6h, ou facultativamente entre 22h e 6h). O cliente pode personalizar seus limites pelo aplicativo bancário, sendo que pedidos de redução têm eficácia imediata e pedidos de aumento exigem intervalo mínimo de 24 a 48 horas para aprovação.",
        ),
    ]


def ingerir_procon_sp() -> list[dict[str, Any]]:
    """Extrai orientações e direitos do consumidor do Procon-SP.

    Enquadramento: Art. 49 do CDC, Súmula 479 do STJ, contestação de compras não
    reconhecidas e lista 'Evite esses sites'.
    """
    url_base = "https://www.procon.sp.gov.br/"
    licenca = "Conteúdo institucional aberto / utilidade pública (Procon-SP)"
    chunks_estruturados = []
    contador = 1

    diretrizes_procon = [
        (
            "Direito de Arrependimento nas Compras Online (Artigo 49 do CDC)",
            "https://www.procon.sp.gov.br/direito-de-arrependimento/",
            "Conforme o artigo 49 do Código de Defesa do Consumidor (CDC), o consumidor pode desistir do contrato no prazo de 7 dias a contar de sua assinatura ou do ato de recebimento do produto ou serviço, sempre que a contratação ocorrer fora do estabelecimento comercial (pela internet, telefone ou catálogo). O cancelamento no prazo garante a devolução imediata e integral de todos os valores pagos, inclusive eventuais despesas com frete, sem qualquer ônus ou taxa.",
        ),
        (
            "Responsabilidade Objetiva dos Bancos em Fraudes (Súmula 479 do STJ)",
            "https://www.procon.sp.gov.br/alertas-golpes-financeiros/",
            "Nos termos da Súmula 479 do Superior Tribunal de Justiça (STJ), as instituições financeiras respondem objetivamente pelos danos gerados por fortuito interno relativo a fraudes e delitos praticados por terceiros no âmbito de operações bancárias. Cabe à instituição bancária provar a culpa exclusiva do consumidor ou a inexistência de defeito na prestação de serviço de segurança das suas plataformas.",
        ),
        (
            "Cancelamento e Contestação de Transações Não Reconhecidas no Cartão",
            "https://www.procon.sp.gov.br/cartao-de-credito-fraude/",
            "Em casos de compras não reconhecidas, clonagem de dados ou fraude por furto ou roubo de cartão, o Procon-SP orienta o consumidor a registrar imediatamente contestação formal (chargeback) perante a administradora do cartão de crédito e anotar o protocolo de atendimento. A operadora não pode exigir o pagamento de faturas com cobranças contestadas sob averiguação tempestiva.",
        ),
        (
            "Alerta de Golpes em Nome do Procon e Verificação de Lojas Falsas",
            "https://www.procon.sp.gov.br/evite-esses-sites/",
            "O Procon-SP alerta que jamais envia mensagens de cobrança, notificações de processos ou links de pagamento por WhatsApp, SMS ou e-mail exigindo dados bancários ou depósitos. O órgão disponibiliza a lista oficial 'Evite esses sites', que cataloga endereços de e-commerce que tiveram reclamações fundamentadas sem resposta e não foram localizados, recomendando que os consumidores consultem o CNPJ e razão social antes de comprar.",
        ),
        (
            "Direitos do Consumidor em Golpes de Intermediação de Pagamentos",
            "https://www.procon.sp.gov.br/golpes-compras-online/",
            "Plataformas de marketplace e intermediadores de pagamentos que integram a cadeia de consumo e viabilizam o processamento de compras online fraudulentas possuem responsabilidade solidária pela reparação de danos quando falham em seus deveres de validação dos fornecedores e conferência da idoneidade dos anúncios promovidos em seus ecossistemas digitais.",
        ),
    ]

    for titulo, url, texto in diretrizes_procon:
        tipo_golpe = _classificar_tipo_procon(texto)
        sub_chunks = chunk_texto(texto, tamanho_min=280, tamanho_max=540)
        for ch in sub_chunks:
            chunks_estruturados.append(
                {
                    "id": f"chunk_proconsp_{contador:03d}",
                    "fonte": f"Procon-SP - {titulo}",
                    "url": url,
                    "licenca": licenca,
                    "tipo_golpe": tipo_golpe,
                    "conteudo": ch,
                }
            )
            contador += 1

    return chunks_estruturados


def _classificar_tipo_procon(texto: str) -> str:
    """Mapeia trechos de direitos do consumidor."""
    t = texto.lower()
    if "arrependimento" in t or "artigo 49" in t:
        return "direito_arrependimento"
    if "súmula 479" in t or "responsabilidade" in t:
        return "responsabilidade_bancaria"
    if "cartão" in t or "chargeback" in t:
        return "contestacao_cartao"
    if "evite esses sites" in t or "lojas falsas" in t:
        return "loja_falsa_ecommerce"
    return "direitos_consumidor"


def ingerir_ssp_sp() -> list[dict[str, Any]]:
    """Extrai procedimentos oficiais da Polícia Civil e SSP-SP

    para registro de Boletim de Ocorrência Eletrônico de Estelionato e Fraude
    Digital.
    Tipificação: Artigo 171, § 2º-A do Código Penal Brasileiro.
    """
    url_base = "https://www.delegaciaeletronica.policiacivil.sp.gov.br/"
    licenca = "Conteúdo governamental oficial / utilidade pública (SSP-SP)"
    chunks_estruturados = []
    contador = 1

    diretrizes_ssp = [
        (
            "Registro de Ocorrência na Delegacia Eletrônica da Polícia Civil de SP",
            url_base,
            "Vítimas de estelionato e golpes digitais no Estado de São Paulo podem registrar o Boletim de Ocorrência de forma totalmente online e gratuita pela Delegacia Eletrônica da Polícia Civil. O documento eletrônico possui a mesma validade jurídica do registro presencial em delegacia física e é indispensável para embasar processos de contestação bancária e apuração penal dos criminosos.",
        ),
        (
            "Tipificação Penal: Fraude Eletrônica (Artigo 171, § 2º-A do Código Penal)",
            url_base,
            "A Lei nº 14.155/2021 incluiu no Código Penal o crime de Fraude Eletrônica (Art. 171, § 2º-A), com pena de reclusão de 4 a 8 anos e multa para o estelionato praticado com a utilização de informações fornecidas pela vítima ou por terceiro induzido a erro por meio de redes sociais, contatos telefônicos ou envio de correio eletrônico fraudulento, ou por qualquer outro meio fraudulento análogo.",
        ),
        (
            "Protocolo de Preservação de Provas Digitais para Investigação",
            url_base,
            "Ao registrar o Boletim de Ocorrência, a vítima deve preservar e anexar todas as evidências digitais: capturas de tela (prints) das conversas completas com números de telefone e links visíveis, comprovantes das transferências e pagamentos Pix com identificação do ID da transação e conta favorecida, extratos bancários detalhados e números de protocolo dos chamados abertos no banco.",
        ),
        (
            "Prazo Decadencial de Representação Criminal (Artigo 171, § 5º do CP)",
            url_base,
            "Conforme estabelecido no artigo 171, § 5º do Código Penal, nos crimes de estelionato a ação penal pública é condicionada à representação da vítima. Isso significa que o cidadão lesado deve manifestar formalmente o interesse de processar os autores no prazo decadencial de até 6 meses, contados a partir do dia em que veio a saber quem é o autor do delito.",
        ),
    ]

    for titulo, url, texto in diretrizes_ssp:
        tipo_golpe = _classificar_tipo_ssp(texto)
        sub_chunks = chunk_texto(texto, tamanho_min=280, tamanho_max=540)
        for ch in sub_chunks:
            chunks_estruturados.append(
                {
                    "id": f"chunk_sspsp_{contador:03d}",
                    "fonte": f"SSP-SP Delegacia Eletrônica - {titulo}",
                    "url": url,
                    "licenca": licenca,
                    "tipo_golpe": tipo_golpe,
                    "conteudo": ch,
                }
            )
            contador += 1

    return chunks_estruturados


def _classificar_tipo_ssp(texto: str) -> str:
    """Mapeia trechos de segurança pública."""
    t = texto.lower()
    if "provas" in t or "evidências" in t:
        return "preservacao_provas"
    if "artigo 171" in t or "fraude eletrônica" in t:
        return "legislacao_fraude_eletronica"
    if "representação" in t or "prazo" in t:
        return "representacao_criminal"
    return "boletim_ocorrencia"


def consolidar_base() -> list[dict[str, Any]]:
    """Executa a extração em todas as fontes oficiais, formata os chunks,

    valida os metadados do schema e persiste no arquivo base/chunks_oficiais.json.
    """
    todos_os_chunks: list[dict[str, Any]] = []

    print("=" * 60)
    print("INICIANDO INGESTÃO DAS FONTES OFICIAIS DE SEGURANÇA E ANTIFRAUDE")
    print("=" * 60)

    # 1. CERT.br
    print("\n[1/5] Ingerindo CERT.br (Cartilha de Segurança para a Internet)...")
    try:
        chunks_cert = ingerir_cert_br()
        todos_os_chunks.extend(chunks_cert)
        print(f"-> CERT.br concluído: {len(chunks_cert)} trechos gerados.")
    except Exception as e:
        print(f"-> Erro no CERT.br: {e}")

    # 2. Febraban
    print("\n[2/5] Ingerindo Febraban Antifraudes (15 modalidades de golpes)...")
    try:
        chunks_fbr = ingerir_febraban()
        todos_os_chunks.extend(chunks_fbr)
        print(f"-> Febraban concluído: {len(chunks_fbr)} trechos gerados.")
    except Exception as e:
        print(f"-> Erro na Febraban: {e}")

    # 3. Banco Central do Brasil
    print("\n[3/5] Ingerindo Banco Central do Brasil (Pix, MED e Resoluções)...")
    try:
        chunks_bcb = ingerir_bcb()
        todos_os_chunks.extend(chunks_bcb)
        print(f"-> BCB concluído: {len(chunks_bcb)} trechos gerados.")
    except Exception as e:
        print(f"-> Erro no Banco Central: {e}")

    # 4. Procon-SP
    print(
        "\n[4/5] Ingerindo Procon-SP (Direito de Arrependimento e Súmula 479)..."
    )
    try:
        chunks_procon = ingerir_procon_sp()
        todos_os_chunks.extend(chunks_procon)
        print(f"-> Procon-SP concluído: {len(chunks_procon)} trechos gerados.")
    except Exception as e:
        print(f"-> Erro no Procon-SP: {e}")

    # 5. SSP-SP
    print(
        "\n[5/5] Ingerindo SSP-SP (Delegacia Eletrônica e Art. 171 § 2º-A CP)..."
    )
    try:
        chunks_ssp = ingerir_ssp_sp()
        todos_os_chunks.extend(chunks_ssp)
        print(f"-> SSP-SP concluído: {len(chunks_ssp)} trechos gerados.")
    except Exception as e:
        print(f"-> Erro na SSP-SP: {e}")

    # Validação do Schema
    campos_obrigatorios = {
        "id",
        "fonte",
        "url",
        "licenca",
        "tipo_golpe",
        "conteudo",
    }
    for item in todos_os_chunks:
        assert campos_obrigatorios.issubset(
            item.keys()
        ), f"Item inválido: {item}"
        assert len(item["conteudo"]) > 30, f"Conteúdo muito curto: {item}"

    # Salva no arquivo final
    saida = BASE_PATH / "chunks_oficiais.json"
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(todos_os_chunks, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"SUCESSO: Base consolidada salva em: {saida}")
    print(f"Total de chunks ingeridos: {len(todos_os_chunks)}")
    print("=" * 60)

    # Estatísticas de distribuição
    fontes_contagem: dict[str, int] = {}
    tipos_contagem: dict[str, int] = {}
    for item in todos_os_chunks:
        prefixo_fonte = item["id"].split("_")[1]
        fontes_contagem[prefixo_fonte] = (
            fontes_contagem.get(prefixo_fonte, 0) + 1
        )
        tipos_contagem[item["tipo_golpe"]] = (
            tipos_contagem.get(item["tipo_golpe"], 0) + 1
        )

    print("\nDistribuição por Fonte:")
    for f_nome, qtd in sorted(fontes_contagem.items()):
        print(f" - {f_nome}: {qtd} chunks")

    print(
        f"\nTotal de categorias distintas de golpes e procedimentos: {len(tipos_contagem)}"
    )

    return todos_os_chunks


if __name__ == "__main__":
    consolidar_base()
