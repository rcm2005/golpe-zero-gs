# Golpe Zero - Auditoria dos 5 Erros Reais

**Disciplina:** Generative AI for Engineering · **Projeto:** Golpe Zero (CitizenTech, proteção 60+ contra golpes digitais)

Cada erro foi observado em execução real do benchmark (`python -m avaliacao.benchmark`, 35 casos). Os números do placar final estão em `avaliacao/placar_benchmark.json`. Os erros 1 a 3 foram corrigidos e remedidos; os erros 4 e 5 seguem abertos, com proposta de correção.

---

## Erro 1 - Viés protetivo: avisos legítimos e conversas comuns classificados como golpe

* **Casos:** 7 legítimos (`tc_016` a `tc_019`, `tc_026` a `tc_028`) e 4 fora de domínio (`tc_021`, `tc_029`, `tc_030`, `tc_031`) saíram como `golpe` em zero-shot e RAG simples.
* **Efeito medido:** acurácia de 68,6% e alucinação de 31,4% nas duas configurações.
* **Causa:** o prompt de sistema mandava "proteger pessoas 60+ contra golpes" e não definia o que é `legitimo` nem `nao_consta`. Sem critério, o modelo de 3B tende a tratar qualquer aviso bancário como suspeito.
* **Correção** (`SYSTEM_PROMPT_BASE` em `src/llm_engine.py`): critérios explícitos, aplicados nesta ordem.
  * `nao_consta`: assunto sem relação com banco, pagamento, compra ou segurança digital.
  * `legitimo`: aviso apenas informativo, sem link, sem telefone novo, sem pedido de pagamento.
  * `golpe`: pede pagar, transferir, clicar, ligar para número da mensagem, informar código, instalar app, ou promete vantagem boa demais.
* **Resultado:** zero-shot foi a 85,7% e a alucinação caiu a 14,3%.

## Erro 2 - Limiar 0,48 descartava golpes reais

* **Casos:** `tc_009` (Correios), `tc_013` (DETRAN) e `tc_014` (loja falsa) recusados com `nao_consta` na Etapa 1.
* **Efeito medido:** acurácia do agente caiu a 60%, abaixo do zero-shot.
* **Causa:** esses tipos de golpe não têm chunk próprio na base e o embedding `all-MiniLM-L6-v2`, treinado sobretudo em inglês, dá score baixo em português (mínimo de golpe 0,3958, abaixo de 0,48). As faixas de score de golpe e de fora de domínio se sobrepõem, então nenhum limiar separa as classes.
* **Correção:** piso vetorial de 0,35 (menor score de golpe menos 0,05) e portão lexical de domínio.

```python
# src/llm_engine.py - Etapa 1 do agente
fora_dominio = not mensagem_no_dominio(mensagem)
if fora_dominio or score_max < threshold:   # threshold = 0.35
    return {"veredito": "nao_consta", ...}  # sem chamar a LLM
```

* **Resultado:** o agente não recusa mais nenhum dos 19 golpes; os 5 casos claramente fora de domínio são recusados em 7 a 315 ms.

## Erro 3 - Instrução de contexto RAG fez o modelo responder `nao_consta` quase sempre

* **Efeito medido:** ao enquadrar o contexto como "material de referência que NÃO indica golpe", RAG e agente erraram 21 de 35 casos (acurácia 40%), quase todos com `nao_consta`, inclusive golpes claros como `tc_002` e `tc_005`.
* **Causa:** a instrução negativa levou o 3B a descartar a mensagem em vez de compará-la com o contexto.
* **Correção:** instrução de comparação em vez de negação.

```text
Compare a mensagem com os trechos: se ela pede ação nos moldes de algum golpe descrito,
o veredito é golpe; se é só um aviso informativo, é legitimo; se o assunto não é
financeiro, é nao_consta.
```

* **Resultado:** RAG e agente foram a 80% (28/35), com fundamentação de 100% e 89,5%.

## Erro 4 - Código de verificação legítimo lido como golpe (aberto)

* **Casos:** `tc_017` e `tc_027` (código de acesso do Caixa Tem, com aviso de não compartilhar) classificados como `golpe` em todas as configurações; `tc_016` (fatura Nubank) e `tc_020` (limite Pix) mudam de veredito quando o contexto de golpes entra no prompt.
* **Causa provável:** o 3B associa "código" e "segurança" a fraude, e os trechos de golpes recuperados reforçam esse viés. A base tem pouquíssimos exemplos de mensagens bancárias legítimas.
* **Proposta:** incluir avisos legítimos no treino LoRA (já presentes no dataset, 48 exemplos) e na base de conhecimento; avaliar se o LoRA reduz o erro (`relatorio/05_lora.md`).

## Erro 5 - Casos adversariais com vocabulário financeiro (aberto)

* **Casos:** `tc_029` (bolo com "urgente"), `tc_030` (pelada com "conta" e "prazo hoje") e `tc_031` (jogo com "conta bloqueada") classificados como `golpe` nas três configurações.
* **Causa:** passam pelo portão lexical por conterem palavras como "conta" e "R$", e o 3B não os reconhece como fora de domínio.
* **Proposta:** exemplos adversariais no treino LoRA; para o produto, manter pessoa no circuito: toda resposta `golpe` orienta confirmar com familiar ou banco por canal oficial antes de qualquer ação, o que limita o dano de um falso positivo.

---

**Nota de método.** O prompt foi ajustado depois de observar os erros 1 a 3 nos mesmos 35 casos de avaliação. Os números finais são, por isso, otimistas para mensagens novas.
