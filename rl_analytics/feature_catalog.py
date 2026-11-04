"""
feature_catalog.py — Catálogo único de metadados, descrições e orientações de features.
Elimina as duplicações de nomes, rótulos de UI, direções de desvio e templates de diagnóstico.
Oferece orientações estruturadas e objetivas: Observação, Possível Interpretação e O que observar no replay.
"""

from typing import Any


FEATURE_CATALOG: dict[str, dict[str, Any]] = {
    "ball_chase_ratio": {
        "display_name": "Proximidade Simultânea à Bola",
        "ui_label": "Proximidade Simultânea à Bola (%)",
        "direction": "above",  # valor alto em relação à média GC+
        "desc": "Proporção de frames com 2 ou mais jogadores a menos de 1500 UU da bola no plano horizontal.",
        "diagnostics": {
            "good": {
                "observation": "Alternância de aproximação à bola alinhada à referência GC+.",
                "interpretation": "A equipe evitou concentração excessiva de jogadores no mesmo setor da bola.",
                "replay_suggestion": "Checar se as transições entre primeiro e segundo homem mantêm fluidez sem hesitação.",
            },
            "neutral": {
                "observation": "Incidência de múltiplos jogadores nas imediações da bola moderadamente próxima à média.",
                "interpretation": "Ocorreram aproximações simultâneas pontuais intercaladas com espaçamento regular.",
                "replay_suggestion": "Verificar se as disputas conjuntas pontuais geraram pressão efetiva ou concorrência na posse.",
            },
            "bad": {
                "observation": "Frequência de múltiplos jogadores próximos da bola superior ao padrão de referência GC+.",
                "interpretation": "Dois ou mais jogadores atuaram com maior frequência no mesmo raio de ação da bola.",
                "replay_suggestion": "Investigar se houve disputas redundantes da mesma bola (double commits) ou pressão ofensiva deliberada.",
            },
        },
        "narratives": {
            "good": "Alternância de aproximação à bola alinhada à referência GC+.",
            "neutral": "Incidência moderada de múltiplos jogadores próximos à bola.",
            "bad": "Frequência de múltiplos jogadores próximos da bola superior à referência GC+; checar no replay se houve disputas redundantes.",
        },
    },
    "defensive_coverage_pct": {
        "display_name": "Cobertura Defensiva Recuada",
        "ui_label": "Cobertura Defensiva (%)",
        "direction": "below",  # valor baixo em relação à média GC+
        "desc": "Proporção de frames com pelo menos um jogador posicionado atrás da linha central no próprio meio-campo.",
        "diagnostics": {
            "good": {
                "observation": "Presença de ao menos um jogador no próprio meio-campo compatível com a referência GC+.",
                "interpretation": "A equipe manteve consistência na retenção de uma peça recuada como segurança territorial.",
                "replay_suggestion": "Avaliar se o jogador na retaguarda manteve bom ângulo de leitura dos lançamentos adversários.",
            },
            "neutral": {
                "observation": "Cobertura territorial recuada em nível intermediário em relação à média de referência.",
                "interpretation": "A equipe alternou entre momentos de cobertura preventiva e avanços coletivos em bloco.",
                "replay_suggestion": "Observar se as subidas em bloco deixaram a retaguarda vulnerável em contra-ataques rápidos.",
            },
            "bad": {
                "observation": "Menor proporção de tempo com jogador recuado na retaguarda em relação ao padrão GC+.",
                "interpretation": "A equipe avançou simultaneamente com todas as peças ao ataque por períodos prolongados.",
                "replay_suggestion": "Verificar se a ausência de terceiro homem recuado gerou vulnerabilidade a contra-ataques diretos.",
            },
        },
        "narratives": {
            "good": "Presença de cobertura territorial no meio defensivo compatível com a referência GC+.",
            "neutral": "Cobertura defensiva recuada intermediária, alternando avanço em bloco e proteção.",
            "bad": "Menor tempo com jogador recuado na retaguarda que a referência GC+; checar velocidade de recomposição no replay.",
        },
    },
    "triangle_area_mean": {
        "display_name": "Área Média de Formação",
        "ui_label": "Área Média do Triângulo (UU²)",
        "direction": "below",
        "desc": "Área média da projeção horizontal (X, Y) do triângulo formado pelos 3 jogadores da equipe.",
        "diagnostics": {
            "good": {
                "observation": "Área média de formação triangular compatível com a distribuição de referência GC+.",
                "interpretation": "A distribuição espacial ampla no terreno de jogo assegurou cobertura de setores variados.",
                "replay_suggestion": "Checar se o espaçamento facilitou opções de passe e cobertura de rebotes na arena.",
            },
            "neutral": {
                "observation": "Área média de formação dentro de faixa intermediária em relação à referência GC+.",
                "interpretation": "Espaçamento médio equilibrado, adaptando-se conforme as fases da posse.",
                "replay_suggestion": "Revisar se o espaçamento atendeu às necessidades tanto de apoio quanto de contenção defensiva.",
            },
            "bad": {
                "observation": "Área média de formação triangular inferior à média do grupo de referência GC+.",
                "interpretation": "Os jogadores atuaram em média mais concentrados geograficamente no plano da arena.",
                "replay_suggestion": "Investigar se o agrupamento refletiu defesa compacta na área ou aglomeração no mesmo setor.",
            },
        },
        "narratives": {
            "good": "Área média de formação ampla e compatível com a referência GC+.",
            "neutral": "Área média de formação em patamar intermediário ao longo da partida.",
            "bad": "Área média de formação mais reduzida que a referência GC+; avaliar se houve aglomeração setorial ou defesa compacta.",
        },
    },
    "triangle_area_std": {
        "display_name": "Variação da Formação",
        "ui_label": "Variação da Área do Triângulo",
        "direction": "above",
        "desc": "Desvio padrão da área do triângulo de formação 2D da equipe ao longo da partida.",
        "diagnostics": {
            "good": {
                "observation": "Estabilidade nas transições de formação triangular entre momentos de expansão e recuo.",
                "interpretation": "Sinaliza transições geométricas graduais e equilibradas ao longo das movimentações coletivas.",
                "replay_suggestion": "Verificar a sincronia da equipe durante recomposições defensivas.",
            },
            "neutral": {
                "observation": "Oscilação moderada na área de formação coletiva durante as transições de posse.",
                "interpretation": "Variação esperada de espaçamento entre fases de pressão ofensiva e transição para defesa.",
                "replay_suggestion": "Observar se a variação de espaçamento acompanhou a dinâmica dos lances.",
            },
            "bad": {
                "observation": "Grande oscilação na área de formação coletiva em relação ao padrão de referência GC+.",
                "interpretation": "Forte amplitude entre momentos de agrupamento próximo e momentos de dispersão aberta.",
                "replay_suggestion": "Verificar se as variações decorreram de rupturas nas recomposições ou de transições rápidas.",
            },
        },
        "narratives": {
            "good": "Variabilidade da área de formação triangular em patamar consistente.",
            "neutral": "Formação geométrica oscila de forma moderada ao longo do jogo.",
            "bad": "Oscilação acentuada na área de formação triangular; revisar recomposições coletivas no replay.",
        },
    },
    "team_spread_mean": {
        "display_name": "Dispersão Média da Equipe",
        "ui_label": "Dispersão Média da Equipe (UU)",
        "direction": "below",
        "desc": "Distância euclidiana média par-a-par entre os jogadores da equipe no plano horizontal.",
        "diagnostics": {
            "good": {
                "observation": "Distância média entre companheiros de equipe equilibrada e alinhada à referência GC+.",
                "interpretation": "Espaçamento par-a-par favorável a linhas de passe e suporte mútuo no terreno.",
                "replay_suggestion": "Observar se a distância facilitou linhas de passe sem isolar jogadores.",
            },
            "neutral": {
                "observation": "Distância média par-a-par entre jogadores dentro da faixa intermediária de referência.",
                "interpretation": "Espaçamento moderado entre companheiros, adaptando-se às dinâmicas de cada lance.",
                "replay_suggestion": "Checar se o posicionamento permitiu apoio imediato ao portador da bola.",
            },
            "bad": {
                "observation": "Distância média entre companheiros inferior à referência GC+.",
                "interpretation": "Os jogadores operaram com proximidade física mais estreita entre si na média dos frames.",
                "replay_suggestion": "Avaliar se a proximidade limitou linhas de passe ou se favoreceu dobras defensivas pontuais.",
            },
        },
        "narratives": {
            "good": "Distância média entre os companheiros de equipe bem distribuída pelo campo.",
            "neutral": "Distribuição par-a-par dentro dos limites médios observados.",
            "bad": "Distância média entre companheiros mais curta que a referência; verificar no replay se houve limitação de opções de passe.",
        },
    },
    "team_spread_std": {
        "display_name": "Variação da Dispersão",
        "ui_label": "Variação da Dispersão (UU)",
        "direction": "above",
        "desc": "Desvio padrão temporal da distância média par-a-par entre os jogadores da equipe.",
        "diagnostics": {
            "good": {
                "observation": "Regularidade no distanciamento mútuo entre companheiros ao longo das fases da partida.",
                "interpretation": "Constância no distanciamento mútuo durante transições e trocas de posse.",
                "replay_suggestion": "Conferir se a equipe manteve conexão posicional mesmo após perdas de posse.",
            },
            "neutral": {
                "observation": "Variação moderada no distanciamento entre colegas durante disputas e transições.",
                "interpretation": "Oscilação esperada nas distâncias durante disputas divididas e transições laterais.",
                "replay_suggestion": "Observar se a oscilação de distâncias ocorreu de forma coordenada.",
            },
            "bad": {
                "observation": "Variabilidade do espaçamento entre jogadores acima da referência GC+.",
                "interpretation": "Alternância frequente entre fases de grande proximidade e momentos de afastamento acentuado.",
                "replay_suggestion": "Checar se ocorreram momentos de afastamento excessivo seguidos de colapso conjunto na bola.",
            },
        },
        "narratives": {
            "good": "Constância de espaçamento entre companheiros mantida ao longo da partida.",
            "neutral": "Variação de distanciamento entre jogadores em nível intermediário.",
            "bad": "Variabilidade de dispersão acima da referência GC+; checar no replay a coordenação de recomposição.",
        },
    },
    "avg_boost_pct": {
        "display_name": "Nível Médio de Boost",
        "ui_label": "Boost Médio Instantâneo (%)",
        "direction": "below",
        "desc": "Nível médio instantâneo de boost armazenado nos tanques dos veículos da equipe (0 a 100%).",
        "diagnostics": {
            "good": {
                "observation": "Nível médio instantâneo de boost nos veículos equilibrado (~48%), compatível com a referência GC+.",
                "interpretation": "Equilíbrio entre consumo contínuo para deslocamento e recolhimento regular de pads.",
                "replay_suggestion": "Verificar se as rotas de rotação priorizaram a coleta de pads pequenos no fluxo do jogo.",
            },
            "neutral": {
                "observation": "Reserva média instantânea de boost em patamar intermediário ao longo da partida.",
                "interpretation": "Padrão de utilização de combustível regular durante as fases do confronto.",
                "replay_suggestion": "Observar se houve escassez de boost em recuperações defensivas rápidas.",
            },
            "bad": {
                "observation": "Nível médio de boost instantâneo abaixo do padrão de referência GC+.",
                "interpretation": "Os veículos operaram com menor quantidade média de combustível armazenada.",
                "replay_suggestion": "Investigar se o menor nível de boost decorreu de rotas de coleta ineficientes ou de pressão contínua sofrida.",
            },
        },
        "narratives": {
            "good": "Reserva média de boost estável e compatível com a referência GC+.",
            "neutral": "Nível médio de boost dentro do padrão normal de consumo e coleta.",
            "bad": "Reserva média instantânea de boost abaixo da referência GC+; verificar rotas de coleta de pads no replay.",
        },
    },
    "avg_dist_to_ball": {
        "display_name": "Distância Média da Bola",
        "ui_label": "Dist. Média até a Bola (UU)",
        "direction": "any",
        "desc": "Distância euclidiana horizontal média de todos os jogadores da equipe até a bola.",
        "diagnostics": {
            "good": {
                "observation": "Distância média coletiva em relação à bola alinhada com a referência GC+.",
                "interpretation": "Equilíbrio entre sustentação de pressão sobre a bola e cobertura preventiva de espaço.",
                "replay_suggestion": "Checar se a equipe sustentou pressão sem descuidar da cobertura de rebotes.",
            },
            "neutral": {
                "observation": "Distância coletiva até a bola em faixa intermediária em relação à referência.",
                "interpretation": "Pequenas oscilações posicionais adaptando-se ao ritmo da partida.",
                "replay_suggestion": "Revisar a velocidade de aproximação à bola nas mudanças de posse.",
            },
            "bad_above": {
                "observation": "Bloco da equipe posicionado em média mais distante da bola que a referência GC+.",
                "interpretation": "O conjunto da equipe posicionou-se mais afastado da bola ao longo dos lances.",
                "replay_suggestion": "Verificar se a maior distância concedeu espaço aos adversários ou refletiu contra-ataque recuado.",
            },
            "bad_below": {
                "observation": "Bloco da equipe posicionado em média mais próximo da bola que a referência GC+.",
                "interpretation": "Posicionamento coletivo bastante adiantado em direção à bola no plano da arena.",
                "replay_suggestion": "Observar se a proximidade gerou vulnerabilidade a lançamentos longos por cima.",
            },
        },
        "narratives": {
            "good": "Posicionamento médio em relação à bola em sintonia com o perfil GC+.",
            "neutral": "Distância média até a bola com oscilação aceitável.",
            "bad_above": "Distância coletiva até a bola acima da referência; checar no replay se houve concessão de espaço para avanços adversários.",
            "bad_below": "Distância coletiva até a bola abaixo da referência; avaliar se a proximidade expôs o time a lançamentos por cima.",
        },
    },
    "min_player_dist_to_ball": {
        "display_name": "Proximidade do 1º Homem",
        "ui_label": "Dist. Mínima à Bola (UU)",
        "direction": "any",
        "desc": "Distância euclidiana horizontal média do jogador mais próximo (primeiro homem) até a bola.",
        "diagnostics": {
            "good": {
                "observation": "Distância do jogador mais próximo até a bola compatível com a referência GC+.",
                "interpretation": "Leitura de tempo de bola apropriada, mantendo espaço de contestação adequado.",
                "replay_suggestion": "Verificar se o primeiro homem conseguiu forçar desarmes ou passes precipitados.",
            },
            "neutral": {
                "observation": "Distância do primeiro homem até a bola dentro de limites razoáveis.",
                "interpretation": "Contestações regulares com momentos pontuais de maior cautela ou ímpeto.",
                "replay_suggestion": "Checar o fechamento de ângulos de chute nas jogadas defensivas.",
            },
            "bad_above": {
                "observation": "Jogador mais próximo manteve maior distância da bola em relação ao padrão GC+.",
                "interpretation": "Contestação menos agressiva ou abordagem tardia ao portador da bola.",
                "replay_suggestion": "Investigar se houve hesitação na abordagem defensiva ou contestação tardia.",
            },
            "bad_below": {
                "observation": "Jogador mais próximo manteve distância muito curta em relação à bola.",
                "interpretation": "Abordagem imediata e contínua sobre a trajetória da bola.",
                "replay_suggestion": "Avaliar se a aproximação excessiva gerou desarmes limpos ou botes precipitados.",
            },
        },
        "narratives": {
            "good": "Jogador mais próximo mantém distância equilibrada para leitura e disputa da bola.",
            "neutral": "Distância do primeiro homem até a bola dentro da faixa média.",
            "bad_above": "Primeiro homem atua distante da bola; verificar no replay se houve hesitação na abordagem defensiva.",
            "bad_below": "Primeiro homem atua muito próximo da bola; checar no replay se houve botes precipitados ou falta de ângulo de leitura.",
        },
    },
    "avg_speed": {
        "display_name": "Velocidade Média (3D)",
        "ui_label": "Velocidade Média (UU/s)",
        "direction": "below",
        "desc": "Velocidade linear média tridimensional (vx, vy, vz) dos veículos da equipe amostrada a cada 30 frames (~1 Hz).",
        "diagnostics": {
            "good": {
                "observation": "Velocidade média tridimensional dinâmica e sustentada, alinhada à referência GC+.",
                "interpretation": "Ritmo ativo de deslocamento e velocidade sustentada de transição pelos jogadores.",
                "replay_suggestion": "Observar se a velocidade foi acompanhada de precisão mecânica nos toques e recuperação de posse.",
            },
            "neutral": {
                "observation": "Velocidade média de deslocamento em patamar intermediário ao longo da partida.",
                "interpretation": "Ritmo moderado, alternando entre momentos de aceleração e pausas posicionais.",
                "replay_suggestion": "Revisar se a equipe acelerou com rapidez nos momentos de transição favorável.",
            },
            "bad": {
                "observation": "Velocidade linear tridimensional média abaixo do padrão de referência GC+.",
                "interpretation": "Ritmo médio de deslocamento mais lento dos veículos durante as fases da partida.",
                "replay_suggestion": "Investigar se a menor velocidade refletiu lentidão nas recomposições ou controle paciente de ritmo.",
            },
        },
        "narratives": {
            "good": "Ritmo de deslocamento e velocidade tridimensional média alinhados à referência GC+.",
            "neutral": "Velocidade média dos veículos em faixa intermediária.",
            "bad": "Velocidade média tridimensional abaixo da referência GC+; investigar no replay se houve hesitação em transições ou controle cauteloso de ritmo.",
        },
    },
}

# Lookup simples para traduções em exibição
FEATURE_LABELS: dict[str, str] = {
    k: v["ui_label"] for k, v in FEATURE_CATALOG.items()
}

# Dicionário compatível com ModelDrivenErrorDetector._FEATURE_DESCRIPTIONS
FEATURE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    k: {
        "name": v["display_name"],
        "direction": v["direction"],
        "desc": v["desc"],
    }
    for k, v in FEATURE_CATALOG.items()
}

# Dicionário compatível com streamlit_app._NARRATIVE_TEMPLATES
NARRATIVE_TEMPLATES: dict[str, dict[str, str]] = {
    k: {
        "direction": v["direction"],
        **v["narratives"],
    }
    for k, v in FEATURE_CATALOG.items()
}


def get_feature_label(key: str) -> str:
    """Retorna o rótulo PT-BR de uma feature ou o próprio nome se não mapeado."""
    return FEATURE_LABELS.get(key, key)


def get_feature_diagnostic_guidance(feat_name: str, z: float, tier: str) -> dict[str, str]:
    """Retorna observação, possível interpretação e sugestão para revisão no replay."""
    cat = FEATURE_CATALOG.get(feat_name, {})
    diag_map = cat.get("diagnostics", {})
    direction = cat.get("direction", "any")

    lookup_key = tier
    if tier == "bad" and direction == "any":
        lookup_key = "bad_above" if z > 0 else "bad_below"

    guidance = diag_map.get(lookup_key) or diag_map.get(tier)
    if not guidance:
        dir_text = "acima" if z > 0 else "abaixo"
        return {
            "observation": f"O valor observado para '{feat_name}' encontra-se {abs(z):.2f}σ {dir_text} da média de referência GC+.",
            "interpretation": "A diferença estatística observada indica variação no padrão em relação ao grupo de referência.",
            "replay_suggestion": "Revisar no replay as jogadas representativas para verificar se o comportamento atendeu às necessidades da partida.",
        }

    return {
        "observation": str(guidance.get("observation", "")),
        "interpretation": str(guidance.get("interpretation", "")),
        "replay_suggestion": str(guidance.get("replay_suggestion", "")),
    }


def compute_z_score(val: float, ref_mean: float, ref_std: float, min_std: float = 0.01) -> float:
    """Calcula o z-score com proteção contra divisão por zero."""
    return (val - ref_mean) / max(ref_std, min_std)


def classify_tier(z: float, direction: str) -> str:
    """Classifica z-score em 'good', 'neutral' ou 'bad' com base na direção."""
    if direction == "above":  # valor alto em relação à referência
        if z > 1.2:
            return "bad"
        if z > 0.5:
            return "neutral"
        return "good"
    elif direction == "below":  # valor baixo em relação à referência
        if z < -1.2:
            return "bad"
        if z < -0.5:
            return "neutral"
        return "good"
    else:  # "any" — qualquer desvio grande em relação à referência
        if abs(z) > 1.2:
            return "bad"
        if abs(z) > 0.5:
            return "neutral"
        return "good"
