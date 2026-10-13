"""
feature_catalog.py — Catálogo único de metadados, descrições e narrativas de features.
Elimina as duplicações de nomes, rótulos de UI, direções de desvio e templates de diagnóstico.
"""

from typing import Any


FEATURE_CATALOG: dict[str, dict[str, Any]] = {
    "ball_chase_ratio": {
        "display_name": "Ball Chasing",
        "ui_label": "Taxa de Ball Chasing",
        "direction": "above",  # valor alto = ruim
        "desc": "Jogadores ficam muito proximos da bola ao mesmo tempo.",
        "narratives": {
            "good": "A disciplina de rotação é boa — os jogadores evitam disputar a bola ao mesmo tempo.",
            "neutral": "A rotação é aceitável, mas há momentos em que mais de um jogador vai na bola ao mesmo tempo.",
            "bad": "O time sofre com <strong>ball chasing</strong>: com frequência, dois ou mais jogadores disputam a bola juntos, deixando o campo desprotegido.",
        },
    },
    "defensive_coverage_pct": {
        "display_name": "Cobertura Defensiva",
        "ui_label": "Cobertura Defensiva (%)",
        "direction": "below",  # valor baixo = ruim
        "desc": "Time fica sem defensores com frequencia.",
        "narratives": {
            "good": "A cobertura defensiva é sólida, com pelo menos um jogador protegendo a retaguarda na maior parte do jogo.",
            "neutral": "A defesa está presente na maioria das jogadas, mas há janelas de vulnerabilidade onde o gol fica exposto.",
            "bad": "O time deixa o gol <strong>desprotegido com frequência</strong>, faltando um jogador recuado para cobrir contra-ataques.",
        },
    },
    "triangle_area_mean": {
        "display_name": "Area do Triangulo",
        "ui_label": "Área Média do Triângulo",
        "direction": "below",
        "desc": "Jogadores ficam muito agrupados, sem cobrir o campo.",
        "narratives": {
            "good": "O espaçamento entre os jogadores é <strong>amplo e eficiente</strong>, cobrindo bastante área do campo.",
            "neutral": "A cobertura de campo está na média, sem se destacar positiva ou negativamente.",
            "bad": "O time fica <strong>muito agrupado</strong>, cobrindo pouca área do campo e facilitando jogadas adversárias.",
        },
    },
    "triangle_area_std": {
        "display_name": "Variacao do Triangulo",
        "ui_label": "Variação da Área do Triângulo",
        "direction": "above",
        "desc": "Formacao do time varia demais entre frames.",
        "narratives": {
            "good": "A formação é <strong>estável</strong> ao longo da partida, indicando rotações bem coordenadas.",
            "neutral": "A formação oscila moderadamente ao longo do jogo, o que é aceitável.",
            "bad": "A formação do time é <strong>instável</strong>: ora muito agrupados, ora muito espalhados, sem um padrão claro de rotação.",
        },
    },
    "team_spread_mean": {
        "display_name": "Dispersao Media",
        "ui_label": "Dispersão Média da Equipe",
        "direction": "below",
        "desc": "Jogadores ficam muito proximos uns dos outros.",
        "narratives": {
            "good": "A distância entre os jogadores é <strong>bem distribuída</strong>, facilitando passes e coberturas.",
            "neutral": "A distribuição dos jogadores pelo campo está dentro do esperado.",
            "bad": "Os jogadores ficam <strong>muito próximos entre si</strong>, limitando as opções de passe e cobertura.",
        },
    },
    "team_spread_std": {
        "display_name": "Rotacao Inconsistente",
        "ui_label": "Variação da Dispersão",
        "direction": "above",
        "desc": "Rotacao inconsistente: hora agrupados, hora dispersos.",
        "narratives": {
            "good": "A rotação é <strong>consistente</strong>, com os jogadores mantendo uma distribuição regular ao longo da partida.",
            "neutral": "A consistência da rotação é razoável, com algumas variações.",
            "bad": "A rotação é <strong>irregular</strong> — há momentos de boa distribuição intercalados com agrupamentos desnecessários.",
        },
    },
    "avg_boost_pct": {
        "display_name": "Gestao de Boost",
        "ui_label": "Média de Boost (%)",
        "direction": "below",
        "desc": "Time fica frequentemente sem boost disponivel.",
        "narratives": {
            "good": "A gestão de boost é <strong>eficiente</strong>, com boas reservas mantidas ao longo da partida.",
            "neutral": "O consumo de boost está dentro do normal, sem excessos nem escassez.",
            "bad": "O time opera com <strong>boost baixo</strong> na maior parte do tempo, limitando a capacidade de reagir rapidamente.",
        },
    },
    "avg_dist_to_ball": {
        "display_name": "Distancia Media da Bola",
        "ui_label": "Dist. Média até a Bola",
        "direction": "any",
        "desc": "Distancia media a bola fora do padrao de rank alto.",
        "narratives": {
            "good": "O posicionamento em relação à bola está <strong>equilibrado</strong>, sem ficar longe nem perto demais.",
            "neutral": "A distância média até a bola está aceitável, mas pode ser otimizada.",
            "bad_above": "O time tende a ficar <strong>longe demais da bola</strong>, perdendo oportunidades de pressão.",
            "bad_below": "O time se posiciona <strong>muito próximo da bola</strong>, reforçando a falta de rotação para trás.",
        },
    },
    "min_player_dist_to_ball": {
        "display_name": "Proximidade Minima da Bola",
        "ui_label": "Dist. Mínima à Bola",
        "direction": "any",
        "desc": "Distancia minima a bola fora do padrao de rank alto.",
        "narratives": {
            "good": "O jogador mais próximo da bola mantém uma <strong>distância saudável</strong>, com espaço para ler e reagir.",
            "neutral": "A distância do jogador mais próximo à bola está dentro do padrão.",
            "bad_above": "O jogador mais próximo fica <strong>distante demais da bola</strong>, demorando a chegar nas disputas.",
            "bad_below": "O jogador mais próximo fica <strong>colado na bola</strong>, sem espaço de reação para mudanças de direção.",
        },
    },
    "avg_speed": {
        "display_name": "Velocidade Media",
        "ui_label": "Velocidade Média",
        "direction": "below",
        "desc": "Velocidade media dos jogadores abaixo do padrao.",
        "narratives": {
            "good": "O ritmo de jogo é <strong>alto</strong>, com movimentação rápida e intencional pelo campo.",
            "neutral": "A velocidade de movimentação está na média.",
            "bad": "O ritmo de jogo está <strong>abaixo do esperado</strong>, com movimentação lenta que dificulta a pressão no adversário.",
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


def compute_z_score(val: float, ref_mean: float, ref_std: float, min_std: float = 0.01) -> float:
    """Calcula o z-score com proteção contra divisão por zero."""
    return (val - ref_mean) / max(ref_std, min_std)


def classify_tier(z: float, direction: str) -> str:
    """Classifica z-score em 'good', 'neutral' ou 'bad' com base na direção."""
    if direction == "above":  # valor alto = ruim
        if z > 1.2:
            return "bad"
        if z > 0.5:
            return "neutral"
        return "good"
    elif direction == "below":  # valor baixo = ruim
        if z < -1.2:
            return "bad"
        if z < -0.5:
            return "neutral"
        return "good"
    else:  # "any" — qualquer desvio grande é ruim
        if abs(z) > 1.2:
            return "bad"
        if abs(z) > 0.5:
            return "neutral"
        return "good"
