"""Módulo de Integração com LLM Local e Pipeline RAG (Fases 1.3 e 1.4).

Comunicação HTTP pura com Ollama local (Qwen 2.5:3B),
saída em JSON com schema fixo e benchmark das 3 arquiteturas:
1. Zero-Shot sem contexto
2. Pipeline RAG Simples
3. Agente Decisor com validação em duas etapas
"""

import json
import re
import time
from typing import Any

import requests
from src.vector_search import MotorBuscaVetorial

OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"
# Piso vetorial da Etapa 1. Escolhido abaixo do menor score de golpe medido
# (0.3958 em avaliacao/recall_busca.json): descartar um golpe real custa mais
# ao idoso do que acionar a LLM em uma mensagem benigna.
THRESHOLD_CORTE = 0.35

# Portao de dominio da Etapa 1: sem nenhum destes termos a mensagem nao trata
# de dinheiro, banco, compra ou seguranca digital e e recusada sem chamar a LLM.
GATILHOS_DOMINIO = re.compile(
    r"pix|banco|banc[aá]ri|cart[aã]o|cr[eé]dito|d[eé]bito|conta|boleto|fatura|r\$|"
    r"transfer|pagamento|pagar|pague|senha|c[oó]digo|token|cpf|chave|limite|"
    r"empr[eé]stimo|investi|aplique|restitui|saldo|compra|parcela|taxa|"
    r"bloque|suspens|leil[aã]o|encomenda|entrega|motoboy|cnh|detran|"
    r"receita federal|inss|aposentad|precat[oó]rio|processo|"
    r"nubank|ita[uú]|bradesco|santander|caixa|banco do brasil|\bbb\b|"
    r"link|http|www\.|\.com|whatsapp|n[uú]mero novo|celular",
    re.IGNORECASE,
)


def mensagem_no_dominio(mensagem: str) -> bool:
    """Retorna True se a mensagem contem algum termo do dominio financeiro."""
    return bool(GATILHOS_DOMINIO.search(mensagem))


SCHEMA_JSON = {
    "veredito": "golpe | legitimo | nao_consta",
    "tipo_golpe": "string (ex: falsa_central, falso_motoboy, nao_se_aplica)",
    "sinais_encontrados": ["lista", "de", "gatilhos", "identificados"],
    "fonte_oficial": "string (ex: Febraban, Banco Central, CERT.br, Procon-SP)",
    "proximo_passo": "string (orientação clara e protetiva voltada a pessoa 60+)"
}

SYSTEM_PROMPT_BASE = f"""Você é o Golpe Zero, assistente especialista em segurança digital voltado para proteger pessoas com mais de 60 anos em São Paulo contra fraudes e estelionatos eletrônicos.

SUAS DIRETRIZES FUNDAMENTAIS:
1. Responda ESTRITAMENTE em formato JSON válido, sem texto antes ou depois, seguindo o schema:
{json.dumps(SCHEMA_JSON, ensure_ascii=False, indent=2)}

2. CRITÉRIOS DE VEREDITO (aplique nesta ordem):
   - "nao_consta": a mensagem não trata de banco, pagamento, compra, crédito ou segurança digital (receitas, clima, esportes, saúde, jogos, rateio de despesas entre amigos). Use tipo_golpe "nao_se_aplica". Urgência ou palavras como "conta" e "prazo" sozinhas NÃO tornam uma mensagem golpe.
   - "legitimo": aviso bancário apenas informativo (compra presencial, fatura, Pix realizado, limite, código de verificação que o próprio usuário pediu) que NÃO pede pagamento, NÃO traz link nem telefone novo e manda usar o aplicativo ou o número do verso do cartão. Use tipo_golpe "nao_se_aplica".
   - "golpe": a mensagem pede que o destinatário pague, transfira, clique em link, ligue para número informado na própria mensagem, informe senha ou código, instale aplicativo ou entregue cartão, sob urgência, ameaça, prêmio ou identidade não verificável; ou promete vantagem boa demais (preço muito abaixo, crédito sem consulta, retorno garantido).
3. A orientação no campo 'proximo_passo' deve ser simples, acolhedora, direta e imperativa (ex: 'NUNCA entregue seu cartão a ninguém', 'DESLIGUE o telefone imediatamente e ligue para o número atrás do seu cartão').
4. Não invente normas nem dados: fundamente-se exclusivamente em diretrizes de fontes oficiais (Febraban, Banco Central, CERT.br, Procon-SP, SSP-SP). Em 'fonte_oficial' cite o nome da instituição da fonte que sustenta o veredito.
"""


class LLMEngineLocal:
    """Cliente HTTP nativo para comunicação com Ollama local."""

    def __init__(self, model_name: str = DEFAULT_MODEL, url: str = OLLAMA_API_URL):
        self.model_name = model_name
        self.url = url

    def gerar_resposta(
        self, prompt_usuario: str, contexto_rag: str = "", temperature: float = 0.1
    ) -> dict[str, Any]:
        """Envia requisição REST ao Ollama forçando saída em formato JSON."""
        prompt_completo = prompt_usuario
        if contexto_rag:
            prompt_completo = (
                f"--- CONTEXTO OFICIAL DE SEGURANÇA (FEBRABAN / BCB / CERT.BR) ---\n"
                f"{contexto_rag}\n"
                f"--- FIM DO CONTEXTO ---\n"
                f"Os trechos acima descrevem golpes conhecidos e orientações oficiais. Compare a mensagem com eles: "
                f"se ela pede ação nos moldes de algum golpe descrito, o veredito é golpe; "
                f"se é só um aviso informativo, é legitimo; se o assunto não é financeiro, é nao_consta.\n\n"
                f"MENSAGEM SUSPEITA RECEBIDA PELO IDOSO:\n"
                f"\"{prompt_usuario}\"\n\n"
                f"Analise a mensagem à luz do contexto oficial acima e devolva o JSON conforme as instruções."
            )

        payload = {
            "model": self.model_name,
            "system": SYSTEM_PROMPT_BASE,
            "prompt": prompt_completo,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
            },
        }

        inicio = time.time()
        try:
            resp = requests.post(self.url, json=payload, timeout=60)
            resp.raise_for_status()
            latencia_segundos = round(time.time() - inicio, 3)

            data = resp.json()
            raw_text = data.get("response", "").strip()

            # Parser JSON com fallback de extração por regex caso haja caracteres extras
            parsed_json = self._fazer_parse_json(raw_text)
            parsed_json["_latencia_seg"] = latencia_segundos
            parsed_json["_modelo"] = self.model_name
            parsed_json["_conformidade_json"] = True
            return parsed_json

        except Exception as e:
            latencia_segundos = round(time.time() - inicio, 3)
            return {
                "veredito": "nao_consta",
                "tipo_golpe": "erro_inferencia",
                "sinais_encontrados": [f"Falha de comunicação: {str(e)}"],
                "fonte_oficial": "Sistema Local",
                "proximo_passo": "Tente novamente ou consulte diretamente seu banco.",
                "_latencia_seg": latencia_segundos,
                "_modelo": self.model_name,
                "_conformidade_json": False,
                "_erro": str(e),
            }

    def _fazer_parse_json(self, raw_text: str) -> dict[str, Any]:
        """Extrai o JSON de JSON mesmo com markdown fences ou quebras."""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Resposta não é um JSON válido: {raw_text[:200]}")


class PipelineGolpeZero:
    """Orquestrador do Benchmark das 3 Configurações Técnicas."""

    def __init__(self, motor: MotorBuscaVetorial | None = None):
        self.motor = motor if motor else MotorBuscaVetorial()
        self.llm = LLMEngineLocal()

    # 1. Configuração Zero-Shot (Sem Contexto RAG)
    def predizer_zero_shot(self, mensagem: str) -> dict[str, Any]:
        """Executa a LLM diretamente sobre a mensagem sem injetar cartilhas oficiais."""
        prompt = f"Analise a seguinte mensagem suspeita: \"{mensagem}\""
        return self.llm.gerar_resposta(prompt_usuario=prompt, contexto_rag="")

    # 2. Configuração RAG Simples
    def predizer_rag_simples(self, mensagem: str, top_k: int = 3) -> dict[str, Any]:
        """Recupera os k chunks mais similares e injeta no prompt da LLM."""
        chunks_recuperados = self.motor.buscar_top_k(mensagem, k=top_k)
        contexto_textos = [
            f"[{c['fonte']}]: {c['conteudo']}" for c in chunks_recuperados
        ]
        contexto_formatado = "\n\n".join(contexto_textos)
        resultado = self.llm.gerar_resposta(
            prompt_usuario=mensagem, contexto_rag=contexto_formatado
        )
        resultado["_chunks_usados"] = [c["id"] for c in chunks_recuperados]
        resultado["_score_maximo"] = chunks_recuperados[0]["score_similaridade"]
        return resultado

    # 3. Configuração Agente Decisor em Duas Etapas
    def predizer_agente_decisor(self, mensagem: str, threshold: float = THRESHOLD_CORTE) -> dict[str, Any]:
        """Etapa 1: Filtro vetorial com limiar de corte empírico.

        Se score < threshold, recusa compulsória com 'não consta' sem acionar LLM.
        Etapa 2: Se score >= threshold, análise RAG fundamentada nas diretrizes oficiais.
        """
        inicio = time.time()
        chunks_recuperados = self.motor.buscar_top_k(mensagem, k=3)
        score_max = chunks_recuperados[0]["score_similaridade"]

        # ETAPA 1: portao de dominio + piso vetorial (compulsorio)
        fora_dominio = not mensagem_no_dominio(mensagem)
        if fora_dominio or score_max < threshold:
            motivo = (
                "Nenhum termo de banco, pagamento ou compra na mensagem"
                if fora_dominio
                else f"Score de similaridade de {score_max:.4f} abaixo do piso ({threshold})"
            )
            duracao = round(time.time() - inicio, 3)
            return {
                "veredito": "nao_consta",
                "tipo_golpe": "nao_se_aplica",
                "sinais_encontrados": [motivo],
                "fonte_oficial": "Base Oficial Febraban/BCB/CERT.br/Procon-SP",
                "proximo_passo": "Mensagem não categorizada como fraude nas bases oficiais conhecidas. Mantenha cautela e contate seu banco se tiver dúvidas.",
                "_latencia_seg": duracao,
                "_modelo": "Filtro Vetorial (Etapa 1)",
                "_conformidade_json": True,
                "_etapa_decisao": "Etapa 1 (Portao de dominio e piso vetorial)",
                "_score_maximo": score_max,
                "_chunks_usados": [],
            }

        # ETAPA 2: Análise RAG Fundamentada pela LLM
        contexto_textos = [
            f"[{c['fonte']}]: {c['conteudo']}" for c in chunks_recuperados
        ]
        contexto_formatado = "\n\n".join(contexto_textos)
        resultado = self.llm.gerar_resposta(
            prompt_usuario=mensagem, contexto_rag=contexto_formatado
        )
        resultado["_etapa_decisao"] = "Etapa 2 (LLM RAG Fundamentado)"
        resultado["_score_maximo"] = score_max
        resultado["_chunks_usados"] = [c["id"] for c in chunks_recuperados]
        return resultado
