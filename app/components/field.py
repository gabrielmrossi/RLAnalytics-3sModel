"""
field.py — Componente de renderização do campo oficial de Rocket League (2D).
"""

import plotly.graph_objects as go

from config import (
    FIELD_HALF_LENGTH,
    FIELD_HALF_WIDTH,
    GOAL_DEPTH,
    GOAL_WIDTH,
)


def draw_field(fig: go.Figure) -> None:
    """Adiciona o traçado do campo oficial no gráfico Plotly (retângulo, meio de campo e traves)."""
    # Retângulo principal do campo
    fig.add_shape(
        type="rect",
        x0=-FIELD_HALF_WIDTH,
        y0=-FIELD_HALF_LENGTH,
        x1=FIELD_HALF_WIDTH,
        y1=FIELD_HALF_LENGTH,
        line=dict(color="#333333", width=2),
        fillcolor="rgba(30, 60, 30, 0.3)",
    )

    # Linha central (meio de campo)
    fig.add_shape(
        type="line",
        x0=-FIELD_HALF_WIDTH,
        y0=0,
        x1=FIELD_HALF_WIDTH,
        y1=0,
        line=dict(color="#444444", width=1, dash="dash"),
    )

    # Círculo central
    fig.add_shape(
        type="circle",
        x0=-1000,
        y0=-1000,
        x1=1000,
        y1=1000,
        line=dict(color="#444444", width=1, dash="dash"),
    )

    # Traves e áreas de gol (Azul: y < 0, Laranja: y > 0)
    for y_pos, color in [
        (-FIELD_HALF_LENGTH, "rgba(52, 152, 219, 0.25)"),
        (FIELD_HALF_LENGTH, "rgba(230, 126, 34, 0.25)"),
    ]:
        sign = 1 if y_pos < 0 else -1
        fig.add_shape(
            type="rect",
            x0=-GOAL_WIDTH / 2,
            y0=y_pos,
            x1=GOAL_WIDTH / 2,
            y1=y_pos + sign * GOAL_DEPTH,
            line=dict(color="#555555", width=1),
            fillcolor=color,
        )
