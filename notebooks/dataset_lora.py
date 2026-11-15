"""Gerador do dataset de treino LoRA do Golpe Zero.

Mensagens sinteticas em portugues (golpes, avisos legitimos e conversas fora de
dominio) montadas a partir de modelos com campos variaveis. Nenhuma mensagem dos
arquivos ``avaliacao/test_cases*.json`` e usada: o gabarito fica 100% held-out.
O dataset e gravado em ``avaliacao/dataset_treino_lora.jsonl``.
"""

import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SAIDA = BASE_DIR / "avaliacao" / "dataset_treino_lora.jsonl"

INSTRUCAO = (
    "Voce e o Golpe Zero, assistente que protege pessoas com mais de 60 anos contra golpes. "
    "Analise a mensagem e responda somente com JSON contendo: veredito (golpe, legitimo ou nao_consta), "
    "tipo_golpe, sinais_encontrados, fonte_oficial e proximo_passo."
)

BANCOS = ["Bradesco", "Santander", "Banco do Brasil", "Itaú", "Caixa", "Nubank", "Inter", "C6 Bank"]
VALORES = ["R$ 480,00", "R$ 1.150,00", "R$ 2.970,00", "R$ 760,00", "R$ 5.300,00", "R$ 389,90", "R$ 1.820,00"]
NOMES = ["Marcos", "Carla", "Rogério", "Fernanda", "Paulo", "Juliana", "Tiago", "Beatriz"]
LINKS = ["bit.ly/3xK9pQ", "seguro-app.xyz/login", "regulariza-cpf.club", "tinyurl.com/promo77", "gov-beneficio.top/entrar"]
FONES = ["0800 591 4410", "(11) 4004-8812", "0300 313 7720", "(11) 93344-5566"]

# tipo -> (fonte_oficial, proximo_passo, sinais, modelos)
GOLPES = {
    "falsa_central": (
        "Febraban Antifraudes",
        "DESLIGUE agora. Ligue para o número que está no verso do seu cartão e nunca informe senha por telefone.",
        ["Ligação ou mensagem em nome do banco", "Número de contato informado na própria mensagem", "Pressa para agir"],
        [
            "{banco} Segurança: detectamos uma movimentação estranha de {valor} na sua conta. Ligue já para {fone} para cancelar.",
            "Aqui é do setor antifraude do {banco}. Sua conta será bloqueada em 1 hora. Retorne para {fone} e informe seus dados.",
            "ALERTA {banco}: débito de {valor} não reconhecido. Se não foi você, fale agora com nossa central no {fone}.",
        ],
    ),
    "falso_whatsapp": (
        "Febraban Antifraudes",
        "NÃO faça o Pix. Ligue para o número antigo do seu familiar ou chame por vídeo para confirmar quem é.",
        ["Familiar com número novo", "Pedido de dinheiro urgente", "Recusa de chamada de voz"],
        [
            "Oi mãe, aqui é o {nome}! Perdi meu celular e este é meu número novo. Preciso que faça um Pix de {valor} hoje, depois explico.",
            "Vó, sou eu, {nome}. Meu cartão bloqueou e preciso pagar um boleto de {valor} agora. Pode me ajudar? Não consigo atender ligação.",
            "Oi pai, troquei de chip. Salva esse contato. Me empresta {valor} por Pix? Devolvo amanhã, é urgente.",
        ],
    ),
    "falso_motoboy": (
        "Febraban Antifraudes",
        "NUNCA entregue cartão ou senha a ninguém. Banco não manda motoboy buscar cartão. Ligue para o número do verso do cartão.",
        ["Retirada de cartão em casa", "Pedido de senha ou chip", "Funcionário não identificado"],
        [
            "{banco}: seu cartão foi clonado. Um motoboy passará aí em 20 minutos para recolher o cartão e a senha.",
            "Informamos que seu cartão final 4471 precisa ser trocado. Entregue ao portador que irá à sua casa, junto com a senha anotada.",
            "Atenção {banco}: corte o cartão e guarde o chip. Nosso entregador irá buscar hoje para a perícia.",
        ],
    ),
    "falso_pix_errado": (
        "Febraban Antifraudes",
        "NÃO devolva o dinheiro por Pix. Procure seu banco e use o Mecanismo Especial de Devolução do Banco Central.",
        ["Alegação de Pix feito por engano", "Pedido de devolução imediata", "Desconhecido pedindo transferência"],
        [
            "Boa noite, fiz um Pix de {valor} para você por engano. Pode devolver na minha chave 11 98877-6655? Estou desesperado.",
            "Desculpe incomodar, transferi {valor} para sua conta errada. Me devolve por favor, é para o tratamento da minha mãe.",
            "Moça, caiu {valor} aí na sua conta por engano. Devolve pra mim agora senão o banco bloqueia tudo.",
        ],
    ),
    "phishing_link": (
        "CERT.br",
        "NÃO clique no link. Abra o aplicativo oficial do banco digitando o endereço você mesmo e apague a mensagem.",
        ["Link encurtado ou domínio estranho", "Pedido de atualização de dados", "Ameaça de bloqueio"],
        [
            "{banco}: seu cadastro expira hoje. Atualize seus dados em {link} para evitar bloqueio da conta.",
            "Você tem pontos {banco} prestes a vencer! Resgate em {link} antes que expirem.",
            "Aviso: acesso não autorizado à sua conta. Confirme sua identidade em {link} em até 2 horas.",
        ],
    ),
    "boleto_falso": (
        "Febraban Antifraudes",
        "NÃO pague. Confira o beneficiário no app do seu banco e peça o boleto novamente pelo site oficial da empresa.",
        ["Boleto recebido por mensagem", "Desconto fora do normal", "Beneficiário diferente da empresa"],
        [
            "Segunda via do seu boleto com desconto de 40% se pagar hoje: valor {valor}. Código de barras no anexo.",
            "Sua fatura venceu e o boleto foi atualizado para {valor}. Pague pelo código enviado aqui para evitar negativação.",
            "Prezado cliente, novo boleto emitido com conta atualizada. Utilize apenas o código abaixo, o anterior foi cancelado.",
        ],
    ),
    "emprestimo_falso": (
        "Febraban Antifraudes",
        "NÃO pague nenhuma taxa antecipada. Empréstimo de verdade não cobra para liberar. Procure seu banco.",
        ["Crédito sem consulta", "Taxa antecipada para liberar", "Contato sem você ter solicitado"],
        [
            "Aposentado, você tem {valor} pré-aprovados sem consulta ao Serasa! Pague a taxa de liberação de R$ 150 por Pix.",
            "Crédito consignado liberado hoje, juros zero. Para receber os {valor} deposite o seguro de R$ 220 antes.",
            "Seu empréstimo de {valor} foi aprovado. Falta apenas a taxa de cadastro. Envie o comprovante para liberarmos.",
        ],
    ),
    "investimento_falso": (
        "CERT.br",
        "NÃO invista. Promessa de lucro garantido é golpe. Confira se a empresa tem registro na CVM antes de qualquer pagamento.",
        ["Retorno garantido", "Pressão para decidir logo", "Pagamento por Pix a desconhecido"],
        [
            "Robô de criptomoedas paga 8% ao dia garantido. Deposite {valor} hoje e saque amanhã. Vagas limitadas!",
            "Grupo VIP de investimentos: triplique seus {valor} em 7 dias. Última chance, entre agora.",
            "Plataforma oficial com lucro de 300% ao mês, sem risco. Comece com {valor} via Pix.",
        ],
    ),
    "restituicao_falsa": (
        "CERT.br",
        "NÃO informe CPF nem dados do cartão. Consulte a restituição apenas no site oficial da Receita Federal.",
        ["Órgão público pedindo dados por mensagem", "Valor a receber como isca", "Link fora do domínio gov.br"],
        [
            "Receita Federal: restituição de {valor} disponível. Confirme seus dados em {link} para receber hoje.",
            "Você tem valores a receber no CPF. Libere {valor} agora pelo link {link}, prazo até meia-noite.",
            "INSS: benefício retido de {valor}. Regularize em {link} para evitar cancelamento.",
        ],
    ),
    "falso_advogado": (
        "Febraban Antifraudes",
        "NÃO pague nada. Procure um advogado de confiança ou a Defensoria Pública e confira o processo no site do tribunal.",
        ["Processo que a pessoa não conhece", "Taxa para liberar dinheiro", "Advogado desconhecido"],
        [
            "Dr. {nome} Advogados: há um alvará de {valor} em seu nome. Para liberar, pague as custas judiciais por Pix hoje.",
            "Seu processo foi ganho! Para receber {valor} precisamos que deposite o imposto de 5% antes da liberação.",
            "Escritório jurídico: o juiz liberou sua indenização de {valor}. Falta pagar a taxa de cartório.",
        ],
    ),
    "leilao_falso": (
        "CERT.br",
        "NÃO pague sinal. Confira o leiloeiro na junta comercial e nunca transfira para conta de pessoa física.",
        ["Preço muito abaixo do mercado", "Pagamento antecipado", "Leiloeiro não verificável"],
        [
            "Leilão judicial de veículos com 70% de desconto. Garanta o lote pagando o sinal de {valor} na chave abaixo.",
            "Moto 0 km por {valor} no leilão da Receita. Reserve hoje por Pix, restam 2 unidades.",
            "Imóveis retomados a partir de {valor}. Envie o sinal agora para segurar seu lote antes do fim.",
        ],
    ),
    "loja_falsa": (
        "Procon-SP",
        "NÃO compre. Pesquise o CNPJ e a reputação da loja antes de pagar e prefira cartão de crédito, que permite contestar.",
        ["Preço irreal", "Pagamento só por Pix", "Site criado recentemente"],
        [
            "Queima total! Geladeira duplex de R$ 5.200 por apenas {valor}, só hoje, pagamento via Pix com mais 10% off.",
            "Celular último modelo por {valor}, estoque acabando. Finalize em {link} e pague por Pix.",
            "Liquidação relâmpago: fogão e micro-ondas por {valor}. Frete grátis, só aceitamos Pix.",
        ],
    ),
    "encomenda_taxa": (
        "CERT.br",
        "NÃO pague a taxa. Rastreie a encomenda apenas no site oficial da transportadora ou dos Correios.",
        ["Encomenda retida", "Taxa pedida por link", "Prazo curto para pagar"],
        [
            "Sua encomenda foi retida. Pague a taxa de {valor} em {link} para liberar a entrega em 24 horas.",
            "Aviso de entrega: tentativa falhou. Reagende pagando {valor} de taxa de redistribuição pelo link.",
            "Pacote internacional aguardando liberação alfandegária. Pague {valor} agora ou será devolvido.",
        ],
    ),
    "multa_cnh": (
        "CERT.br",
        "NÃO clique. Consulte multas e CNH somente no site ou aplicativo oficial do órgão de trânsito.",
        ["Órgão de trânsito por mensagem", "Ameaça de suspensão", "Link fora do domínio oficial"],
        [
            "Trânsito: você tem multa pendente de {valor}. Evite a suspensão da CNH regularizando em {link}.",
            "AVISO: pontuação excedida. Pague {valor} no link {link} hoje para não perder a habilitação.",
            "Sua CNH será cassada. Quite o débito de {valor} agora pelo endereço informado.",
        ],
    ),
    "acesso_remoto": (
        "Febraban Antifraudes",
        "NÃO instale nada e NÃO passe códigos. Encerre a conversa e procure seu banco pelo número oficial.",
        ["Pedido para instalar aplicativo", "Controle remoto do celular", "Pedido de códigos recebidos por SMS"],
        [
            "Suporte {banco}: para corrigir sua conta instale o aplicativo de suporte no link {link} e me informe o código.",
            "Para cancelar a cobrança de {valor} precisamos acessar seu celular. Baixe o programa e passe o código que aparecer.",
            "Atualização obrigatória do app do {banco}. Instale a versão do link e informe o código de ativação enviado por SMS.",
        ],
    ),
    "maquininha_falsa": (
        "Febraban Antifraudes",
        "NÃO aproxime nem digite a senha. Peça outra forma de pagamento e confira o valor na máquina antes de pagar.",
        ["Máquina supostamente quebrada", "Pedido de senha em aparelho alheio", "Cobrança de valor diferente"],
        [
            "Minha maquininha quebrou. Pode passar o seu cartão na minha e me falar a senha para eu digitar? São {valor}.",
            "Senhora, o leitor está com defeito, me empresta o cartão para testar na outra máquina com a senha.",
            "O entregador avisou que a máquina só funciona se a senha for informada em voz alta. Total de {valor}.",
        ],
    ),
}

LEGITIMOS = [
    ("Banco do Brasil: compra aprovada de {valor} na loja Americanas, cartão final 2231, em {data}. Detalhes no app BB.", "Procon-SP"),
    ("Nubank: sua fatura de {valor} vence em {data}. Consulte o app para ver os lançamentos. Não pedimos senha por mensagem.", "Banco Central do Brasil"),
    ("Itaú: Pix de {valor} enviado com sucesso para Mercado do Bairro em {data}. Qualquer dúvida, use o app.", "Banco Central do Brasil"),
    ("Caixa: seu código de acesso é {cod}. Não compartilhe com ninguém, nem com funcionários do banco.", "Febraban Antifraudes"),
    ("Santander: o limite de Pix noturno da sua conta é {valor}. Para alterar, acesse o aplicativo oficial.", "Banco Central do Brasil"),
    ("Bradesco: saque de {valor} realizado no caixa eletrônico em {data}. Se não reconhece, ligue para o número do verso do cartão.", "Febraban Antifraudes"),
    ("Inter: recebemos seu pagamento de {valor} referente ao boleto da escola. Obrigado.", "Banco Central do Brasil"),
    ("C6 Bank: compra presencial aprovada de {valor} na farmácia, cartão final 8840. Veja o extrato no app.", "Procon-SP"),
]

FORA_DOMINIO = [
    "Vovó, qual a temperatura ideal para assar pão de queijo? Quero fazer para o lanche da tarde.",
    "Alguém sabe a que horas passa o ônibus 175 no ponto da Rua Augusta?",
    "Preciso de uma receita simples de sopa de legumes para o jantar de hoje.",
    "O jogo do São Paulo foi remarcado para domingo às 16h, vamos ao estádio?",
    "Dona Maria, a consulta com o cardiologista está confirmada para quinta de manhã no posto de saúde.",
    "Como faço para plantar manjericão em vaso na varanda do apartamento?",
    "Vai chover no fim de semana? Estou pensando em ir à praia com as crianças.",
    "Me indica um filme leve para assistir hoje à noite com a família?",
    "A reunião do condomínio será sábado às 10h no salão de festas, tragam sugestões.",
    "Qual o melhor jeito de tirar mancha de gordura de camisa branca?",
    "Parabéns pelo aniversário, vó! Te amamos muito, vamos almoçar juntos amanhã.",
    "Quem sabe o telefone da padaria que entrega pão quentinho na Vila Prudente?",
    "A aula de hidroginástica começa às 8h, não esquece a touca e a toalha.",
    "Meu neto passou no vestibular! Estamos todos muito felizes, vamos comemorar no domingo.",
]


def _preencher(modelo: str, rng: random.Random) -> str:
    return modelo.format(
        banco=rng.choice(BANCOS), valor=rng.choice(VALORES), nome=rng.choice(NOMES),
        link=rng.choice(LINKS), fone=rng.choice(FONES),
        data=f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}", cod=f"{rng.randint(100, 999)}.{rng.randint(100, 999)}",
    )


def _saida(veredito: str, tipo: str, sinais: list[str], fonte: str, passo: str) -> str:
    return json.dumps(
        {"veredito": veredito, "tipo_golpe": tipo, "sinais_encontrados": sinais, "fonte_oficial": fonte, "proximo_passo": passo},
        ensure_ascii=False,
    )


def gerar_dataset(semente: int = 7, por_tipo: int = 9) -> list[dict[str, str]]:
    """Gera ~240 exemplos (golpes, legitimos e fora de dominio) com semente fixa."""
    rng = random.Random(semente)
    exemplos: list[dict[str, str]] = []
    for tipo, (fonte, passo, sinais, modelos) in GOLPES.items():
        for i in range(por_tipo):
            msg = _preencher(modelos[i % len(modelos)], rng)
            exemplos.append({"instruction": INSTRUCAO, "input": msg, "output": _saida("golpe", tipo, sinais, fonte, passo)})
    for i in range(48):
        modelo, fonte = LEGITIMOS[i % len(LEGITIMOS)]
        msg = _preencher(modelo, rng)
        exemplos.append({
            "instruction": INSTRUCAO, "input": msg,
            "output": _saida("legitimo", "nao_se_aplica",
                             ["Aviso apenas informativo", "Sem link nem telefone novo", "Orienta usar canal oficial"],
                             fonte, "Mensagem normal. Confira o lançamento pelo aplicativo oficial do seu banco."),
        })
    for i in range(48):
        msg = FORA_DOMINIO[i % len(FORA_DOMINIO)]
        exemplos.append({
            "instruction": INSTRUCAO, "input": msg,
            "output": _saida("nao_consta", "nao_se_aplica",
                             ["Assunto sem relação com banco, pagamento ou compra"],
                             "Sistema Golpe Zero", "Esta mensagem não trata de golpe financeiro. Pode responder normalmente."),
        })
    rng.shuffle(exemplos)
    return exemplos


def verificar_vazamento(exemplos: list[dict[str, str]]) -> int:
    """Conta mensagens de treino identicas a alguma do gabarito (deve ser 0)."""
    teste: set[str] = set()
    for nome in ("test_cases", "test_cases_stress"):
        with open(BASE_DIR / "avaliacao" / f"{nome}.json", encoding="utf-8") as f:
            teste |= {t["mensagem_recebida"].strip().lower() for t in json.load(f)}
    return sum(e["input"].strip().lower() in teste for e in exemplos)


if __name__ == "__main__":
    dados = gerar_dataset()
    assert verificar_vazamento(dados) == 0
    with open(SAIDA, "w", encoding="utf-8") as f:
        for d in dados:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(dados)} exemplos gravados em {SAIDA}")
