"""Shared design tokens and helpers for the CCO-XAI dashboard."""

from __future__ import annotations

import streamlit as st

COLORS = {
    "primary": "#8C1D24",
    "ink": "#14233C",
    "text": "#1F2937",
    "muted": "#6B7280",
    "border": "#E5E7EB",
    "surface": "#F6F7F9",
    "teal": "#2F6F6B",
    "sand": "#A98F6F",
}

COLORWAY = [COLORS["primary"], COLORS["ink"], COLORS["teal"], COLORS["sand"]]

CORR_SCALE = [
    [0.0, "#8FA6BC"],
    [0.5, "#FFFFFF"],
    [1.0, "#C98F93"],
]

DIVERGING_SCALE = [
    [0.0, COLORS["ink"]],
    [0.5, "#F1F2F4"],
    [1.0, COLORS["primary"]],
]

SEQUENTIAL_SCALE = [
    [0.0, "#E3EAE9"],
    [0.5, COLORS["teal"]],
    [1.0, COLORS["primary"]],
]

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
}

LABEL_HINT = (
    "Formato: nome da coluna do CSV (símbolo da grandeza), "
    "por exemplo volume_mm3 (V_intra)."
)

LABELS = {
    "deltaP_mmHg": "deltaP_mmHg (ΔP)",
    "gamma": "gamma (γ)",
    "Q_ml_min": "Q_ml_min (Q_perf)",
    "Nterm": "Nterm (N_term)",
    "volume_mm3": "volume_mm3 (V_intra)",
    "raio_root_mm": "raio_root_mm (r_root)",
    "id_configuracao": "Configuração",
}

TARGET_LABELS = {
    "volume_mm3": "volume_mm3 (V_intra)",
    "raio_root_mm": "raio_root_mm (r_root)",
}

TARGET_UNITS = {"volume_mm3": "mm³", "raio_root_mm": "mm"}


GLOBAL_CSS = """
<style>
    .block-container {
        max-width: 1240px;
        padding-top: 2.4rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #FAFAFB;
        border-right: 1px solid #E6E8EC;
    }
    [data-testid="stSidebarNav"] a {
        border-radius: 8px;
    }
    [data-testid="stMetric"] {
        background: #F7F8FA;
        border: 1px solid #E6E8EC;
        border-radius: 10px;
        padding: 0.9rem 1rem;
    }
    [data-testid="stMetricLabel"] p {
        color: #6B7280;
        font-size: 0.78rem;
        letter-spacing: 0.01em;
    }
    [data-testid="stMetricValue"] {
        color: #14233C;
        font-size: 1.45rem;
    }
    h2 {
        letter-spacing: -0.015em;
        margin-bottom: 0.15rem;
    }
    h3, h4 {
        letter-spacing: -0.01em;
    }
    .cco-brand {
        color: #14233C;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .cco-brand span {
        color: #8C1D24;
    }
    .cco-tagline {
        color: #6B7280;
        font-size: 0.76rem;
        margin-top: -0.35rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }
</style>
"""


def apply_global_style() -> None:
    """Inject the shared CSS once, from the app entrypoint."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str | None = None, icon: str | None = None) -> None:
    """Render a consistent page heading with an optional Material icon."""
    prefix = f":material/{icon}: " if icon else ""
    st.markdown(f"## {prefix}{title}")
    if subtitle:
        st.caption(subtitle)


def plotly_theme(fig, height: int | None = None, showlegend: bool | None = None):
    """Apply the dashboard's restrained Plotly theme in place."""
    fig.update_layout(
        template="plotly_white",
        colorway=COLORWAY,
        font=dict(
            family="Source Sans Pro, Segoe UI, sans-serif",
            size=12,
            color=COLORS["text"],
        ),
        title=dict(
            font=dict(size=14, color=COLORS["ink"]),
            x=0,
            xanchor="left",
        ),
        margin=dict(l=10, r=10, t=84, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=COLORS["border"],
            font_color=COLORS["text"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=None,
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=COLORS["border"],
        ticks="outside",
        tickcolor=COLORS["border"],
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["border"],
        gridwidth=0.6,
        zeroline=False,
        linecolor=COLORS["border"],
        ticks="outside",
        tickcolor=COLORS["border"],
    )
    if height:
        fig.update_layout(height=height)
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return fig
