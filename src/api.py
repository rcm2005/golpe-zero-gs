"""API REST FastAPI para o Sistema Golpe Zero (CitizenTech).

Permite que aplicações frontend (mobile, web, WhatsApp bot) consumam:
1. Análise de texto de mensagens suspeitas via Agente Decisor em 2 etapas.
2. Análise de imagens e capturas de tela (prints).
3. Busca vetorial direta nas cartilhas oficiais (Febraban, BCB, CERT.br, Procon-SP).
4. Status e monitoramento do motor local Ollama e da base vetorial.
"""

import time
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.llm_engine import PipelineGolpeZero, THRESHOLD_CORTE
from src.vector_search import MotorBuscaVetorial
from vision.inference import DetectorVisualGolpe

# Inicializa aplicação FastAPI com documentação Swagger automática
app = FastAPI(
    title="Golpe Zero API — CitizenTech 2026.2",
    description="API de proteção de idosos 60+ contra estelionatos digitais e fraudes eletrônicas em SP.",
    version="1.0.0",
)

# Habilita CORS irrestrito para consumo direto por apps mobile (React Native, Flutter, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa motores em memória (singleton)
motor_vetorial: Optional[MotorBuscaVetorial] = None
pipeline_ia: Optional[PipelineGolpeZero] = None
detector_visual: Optional[DetectorVisualGolpe] = None


@app.on_event("startup")
def inicializar_motores():
    """Carrega o motor vetorial, detector visual YOLO e o pipeline na inicialização."""
    global motor_vetorial, pipeline_ia, detector_visual
    print("[API] Inicializando motor vetorial e pipeline de IA...")
    motor_vetorial = MotorBuscaVetorial()
    pipeline_ia = PipelineGolpeZero(motor=motor_vetorial)
    print("[API] Inicializando detector visual YOLOv8n...")
    try:
        detector_visual = DetectorVisualGolpe()
        print("[API] Detector visual YOLO inicializado!")
    except Exception as e:
        print(f"[API] Aviso: Módulo visual operando em modo fallback ({e})")
        detector_visual = None
    print("[API] Motores inicializados com sucesso!")


class AnaliseTextoRequest(BaseModel):
    texto: str = Field(..., description="Texto da mensagem recebida no WhatsApp, SMS ou ligação transcrita.")
    arquitetura: str = Field(
        default="agente",
        description="Arquitetura de inferência: 'agente' (2 etapas), 'rag' (RAG simples), ou 'zero_shot' (direto na LLM)."
    )
    threshold: float = Field(
        default=THRESHOLD_CORTE,
        description="Limiar de corte empírico para o filtro vetorial do Agente Decisor."
    )


class AnaliseResponse(BaseModel):
    veredito: str = Field(..., description="'golpe', 'legitimo' ou 'nao_consta'")
    tipo_golpe: str = Field(..., description="Classificação do golpe ou 'nao_se_aplica'")
    sinais_encontrados: list[str] = Field(..., description="Gatilhos e evidências identificadas na mensagem")
    fonte_oficial: str = Field(..., description="Norma ou cartilha oficial utilizada como fundamento")
    proximo_passo: str = Field(..., description="Instrução direta, clara e protetiva em linguagem voltada para 60+")
    metadados: dict[str, Any] = Field(default_factory=dict, description="Métricas de latência, modelo e score vetorial")


@app.get("/api/health", tags=["Monitoramento"])
def health_check():
    """Verifica a integridade dos serviços locais (API, vetorizador e Ollama)."""
    import requests
    ollama_ok = False
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    return {
        "status": "online",
        "timestamp": time.time(),
        "motor_vetorial_carregado": motor_vetorial is not None,
        "total_chunks_oficiais": len(motor_vetorial.chunks) if motor_vetorial else 0,
        "ollama_local_ativo": ollama_ok,
        "modelo_padrao": "qwen2.5:3b",
    }


@app.post("/api/analisar/texto", response_model=AnaliseResponse, tags=["Análise de Mensagens"])
def analisar_texto(req: AnaliseTextoRequest):
    """Analisa uma mensagem de texto suspeita e devolve o veredito estruturado."""
    if not pipeline_ia:
        raise HTTPException(status_code=503, detail="Pipeline de IA ainda não inicializado.")

    texto_limpo = req.texto.strip()
    if not texto_limpo:
        raise HTTPException(status_code=400, detail="O texto da mensagem não pode ser vazio.")

    # Executa a arquitetura escolhida
    if req.arquitetura == "zero_shot":
        res = pipeline_ia.predizer_zero_shot(texto_limpo)
    elif req.arquitetura == "rag":
        res = pipeline_ia.predizer_rag_simples(texto_limpo, top_k=3)
    else:
        res = pipeline_ia.predizer_agente_decisor(texto_limpo, threshold=req.threshold)

    # Separa metadados internos da resposta do contrato
    metadados = {
        "latencia_seg": res.get("_latencia_seg", 0.0),
        "modelo": res.get("_modelo", "desconhecido"),
        "conformidade_json": res.get("_conformidade_json", True),
        "score_maximo": res.get("_score_maximo", 0.0),
        "chunks_usados": res.get("_chunks_usados", []),
        "etapa_decisao": res.get("_etapa_decisao", "N/A"),
    }

    return AnaliseResponse(
        veredito=res.get("veredito", "nao_consta"),
        tipo_golpe=res.get("tipo_golpe", "nao_se_aplica"),
        sinais_encontrados=res.get("sinais_encontrados", []),
        fonte_oficial=res.get("fonte_oficial", "Não identificada"),
        proximo_passo=res.get("proximo_passo", "Mantenha cautela e consulte seu banco."),
        metadados=metadados,
    )


@app.post("/api/analisar/imagem", response_model=AnaliseResponse, tags=["Análise de Mensagens"])
async def analisar_imagem(
    foto: UploadFile = File(..., description="Captura de tela ou foto da mensagem/comprovante suspeito"),
    texto_adicional: Optional[str] = Form(default="", description="Texto extra ou contexto digitado pelo idoso")
):
    """Recebe print de celular e integra visão computacional YOLOv8 + Agente Decisor LLM."""
    if not pipeline_ia:
        raise HTTPException(status_code=503, detail="Pipeline de IA ainda não inicializado.")

    # Lê arquivo da imagem
    conteudo_bytes = await foto.read()
    if len(conteudo_bytes) == 0:
        raise HTTPException(status_code=400, detail="Arquivo de imagem vazio.")

    # 1. Inferência com o modelo YOLO treinado nas 5 classes
    resultado_visao = {
        "elementos_detectados": [],
        "resumo_texto": "Módulo visual não ativo",
        "nivel_risco_visual": "INDETERMINADO",
        "total_elementos": 0,
        "contagem_classes": {},
        "caminho_imagem_anotada": None,
    }
    if detector_visual:
        try:
            nome_saida = f"analise_{int(time.time())}_{foto.filename}"
            resultado_visao = detector_visual.analisar_imagem(
                conteudo_bytes,
                salvar_anotada=True,
                nome_saida=nome_saida
            )
        except Exception as err:
            print(f"[API] Erro na inferência visual YOLO: {err}")

    # 2. Constrói contexto multimodal para o Agente Decisor
    partes_contexto = [
        f"Captura de tela enviada para análise: {foto.filename}",
        f"Evidências visuais identificadas pelo modelo YOLO: {resultado_visao.get('resumo_texto', '')}",
        f"Nível de risco visual estimado: {resultado_visao.get('nivel_risco_visual', 'INDETERMINADO')}"
    ]
    if texto_adicional and texto_adicional.strip():
        partes_contexto.append(f"Contexto fornecido pelo usuário: {texto_adicional.strip()}")

    texto_para_analise = ". ".join(partes_contexto)

    # 3. Executa o Agente Decisor (com threshold ajustado para análise multimodal)
    res = pipeline_ia.predizer_agente_decisor(texto_para_analise, threshold=THRESHOLD_CORTE)

    # Consolida sinais detectados visuais e textuais
    sinais_finais = list(res.get("sinais_encontrados", []))
    for el in resultado_visao.get("elementos_detectados", []):
        sinais_finais.append(f"Elemento Visual ({el['classe']}): Confiança {el['confianca']*100:.1f}%")

    metadados = {
        "latencia_seg": res.get("_latencia_seg", 0.0),
        "modelo_texto": res.get("_modelo", "desconhecido"),
        "modelo_visao": "YOLOv8n-GolpeZero",
        "arquivo_recebido": foto.filename,
        "tamanho_bytes": len(conteudo_bytes),
        "score_maximo_vetorial": res.get("_score_maximo", 0.0),
        "nivel_risco_visual": resultado_visao.get("nivel_risco_visual", "INDETERMINADO"),
        "total_elementos_visuais": resultado_visao.get("total_elementos", 0),
        "contagem_classes_visuais": resultado_visao.get("contagem_classes", {}),
        "caminho_imagem_anotada": resultado_visao.get("caminho_imagem_anotada", None),
    }

    return AnaliseResponse(
        veredito=res.get("veredito", "nao_consta"),
        tipo_golpe=res.get("tipo_golpe", "nao_se_aplica"),
        sinais_encontrados=sinais_finais if sinais_finais else ["Elemento gráfico em análise"],
        fonte_oficial=res.get("fonte_oficial", "Base Febraban / CERT.br"),
        proximo_passo=res.get("proximo_passo", "Não clique em links do print e confira no app oficial do banco."),
        metadados=metadados,
    )


@app.get("/api/buscar-chunks", tags=["Base de Conhecimento"])
def buscar_chunks_endpoint(q: str, k: int = 3):
    """Realiza busca vetorial manual direta para depuração ou consulta do app."""
    if not motor_vetorial:
        raise HTTPException(status_code=503, detail="Motor vetorial não inicializado.")
    
    resultados = motor_vetorial.buscar_top_k(q, k=k)
    return {
        "consulta": q,
        "total_retornados": len(resultados),
        "chunks": resultados,
    }
