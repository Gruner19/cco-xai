import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ui import (
    COLORS,
    LABEL_HINT,
    LABELS,
    PLOTLY_CONFIG,
    page_header,
    plotly_theme,
)

page_header(
    "Predictor",
    "Ajuste os parâmetros do CCO e consulte as previsões dos modelos treinados.",
    icon="tune",
)
st.caption(LABEL_HINT)

base_dir = Path(__file__).resolve().parents[2]
models_dir = base_dir / "modelos"
features = ["deltaP_mmHg", "gamma", "Q_ml_min", "Nterm"]


@st.cache_data
def carregar_dados_referencia():
    candidatos = [
        base_dir / "dados" / "banco_CCO_XAI_final_500_arvores.csv",
        base_dir / "banco_CCO_XAI_final_500_arvores.csv",
    ]
    for caminho in candidatos:
        if caminho.exists():
            return pd.read_csv(caminho)
    return pd.DataFrame()


df_ref = carregar_dados_referencia()

MODELOS = {
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Árvore de Decisão": "decision_tree",
    "Regressão Linear": "linear_regression",
    "Dummy (Média)": "dummy",
}

with st.sidebar:
    st.markdown("### :material/settings: Modelo")
    modelo_nome = st.selectbox("Algoritmo preditor", list(MODELOS.keys()), index=0)
    st.caption(
        "Os modelos compartilham a mesma divisão agrupada por configuração "
        "e foram ajustados apenas no conjunto de desenvolvimento."
    )

modelo_id = MODELOS[modelo_nome]


@st.cache_resource
def carregar_modelo(mod_id: str, target: str):
    caminho = models_dir / f"{mod_id}_{target}.joblib"
    if caminho.exists():
        return joblib.load(caminho)
    return None


mod_vol = carregar_modelo(modelo_id, "volume_mm3")
mod_raio = carregar_modelo(modelo_id, "raio_root_mm")

if mod_vol is None or mod_raio is None:
    st.warning(
        "Os modelos treinados ainda não estão disponíveis. Execute "
        "`python scripts/02_modelos.py` e recarregue a página.",
        icon=":material/hourglass_empty:",
    )
    st.stop()

with st.container(border=True):
    st.markdown("#### :material/tune: Parâmetros de entrada")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        deltaP = st.slider(
            LABELS["deltaP_mmHg"],
            min_value=20.0,
            max_value=40.0,
            value=30.0,
            step=0.5,
            help="Queda de pressão ΔP, em mmHg.",
        )
    with col2:
        gamma = st.slider(
            LABELS["gamma"],
            min_value=2.10,
            max_value=3.00,
            value=2.55,
            step=0.01,
            help="Expoente de bifurcação de Murray γ, adimensional.",
        )
    with col3:
        Q_perf = st.slider(
            LABELS["Q_ml_min"],
            min_value=200.0,
            max_value=700.0,
            value=500.0,
            step=10.0,
            help="Vazão total de perfusão Q_perf, em mL/min.",
        )
    with col4:
        Nterm = st.slider(
            LABELS["Nterm"],
            min_value=50,
            max_value=400,
            value=250,
            step=25,
            help="Número de segmentos terminais N_term.",
        )

X_input = pd.DataFrame(
    [
        {
            "deltaP_mmHg": deltaP,
            "gamma": gamma,
            "Q_ml_min": Q_perf,
            "Nterm": Nterm,
        }
    ]
)

pred_vol = float(mod_vol.predict(X_input)[0])
pred_raio = float(mod_raio.predict(X_input)[0])

media_vol = float(df_ref["volume_mm3"].mean()) if not df_ref.empty else None
media_raio = float(df_ref["raio_root_mm"].mean()) if not df_ref.empty else None

st.markdown("#### :material/insights: Resultado da previsão")
res_col1, res_col2 = st.columns(2)

with res_col1:
    st.metric(
        f"{LABELS['volume_mm3']} previsto",
        f"{pred_vol:.2f} mm³",
        delta=(
            f"{pred_vol - media_vol:+.2f} mm³ vs. média global"
            if media_vol is not None
            else None
        ),
    )

with res_col2:
    st.metric(
        f"{LABELS['raio_root_mm']} previsto",
        f"{pred_raio:.3f} mm",
        delta=(
            f"{pred_raio - media_raio:+.3f} mm vs. média global"
            if media_raio is not None
            else None
        ),
    )

st.caption(
    f"Modelo selecionado: **{modelo_nome}** · "
    "Previsões calculadas com as quatro entradas do CCO."
)

with st.expander("Ver entrada enviada ao modelo"):
    df_entrada = pd.DataFrame(
        {
            "Variável": [LABELS[f] for f in features],
            "Valor": [deltaP, gamma, Q_perf, Nterm],
        }
    )
    st.dataframe(df_entrada, hide_index=True, width="stretch")

if not df_ref.empty:
    st.markdown("#### :material/bar_chart: Posição no banco de referência")
    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        fig_pv = px.histogram(
            df_ref,
            x="volume_mm3",
            nbins=32,
            labels={"volume_mm3": LABELS["volume_mm3"]},
            title=f"{LABELS['volume_mm3']} — volume intravascular (mm³)",
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig_pv.update_traces(marker_line_color="white", marker_line_width=0.5, opacity=0.85)
        fig_pv.add_vline(
            x=pred_vol,
            line_color=COLORS["ink"],
            line_dash="dash",
            annotation_text=f"previsto: {pred_vol:.1f}",
            annotation_position="top",
        )
        fig_pv = plotly_theme(fig_pv, height=340, showlegend=False)
        st.plotly_chart(fig_pv, width="stretch", config=PLOTLY_CONFIG, theme=None)

    with dist_col2:
        fig_pr = px.histogram(
            df_ref,
            x="raio_root_mm",
            nbins=32,
            labels={"raio_root_mm": LABELS["raio_root_mm"]},
            title=f"{LABELS['raio_root_mm']} — raio proximal da raiz (mm)",
            color_discrete_sequence=[COLORS["teal"]],
        )
        fig_pr.update_traces(marker_line_color="white", marker_line_width=0.5, opacity=0.85)
        fig_pr.add_vline(
            x=pred_raio,
            line_color=COLORS["ink"],
            line_dash="dash",
            annotation_text=f"previsto: {pred_raio:.3f}",
            annotation_position="top",
        )
        fig_pr = plotly_theme(fig_pr, height=340, showlegend=False)
        st.plotly_chart(fig_pr, width="stretch", config=PLOTLY_CONFIG, theme=None)

st.markdown("#### :material/show_chart: Sensibilidade local")
st.caption(
    "Cada curva varia uma entrada e mantém as demais fixas no valor atual. "
    "A linha vertical marca o ponto de operação selecionado."
)

sens_feat = st.selectbox(
    "Entrada a variar",
    features,
    index=2,
    format_func=lambda f: LABELS[f],
)

RANGES = {
    "deltaP_mmHg": (20.0, 40.0),
    "gamma": (2.10, 3.00),
    "Q_ml_min": (200.0, 700.0),
    "Nterm": (50.0, 400.0),
}
if not df_ref.empty and df_ref[sens_feat].nunique() > 1:
    lo = float(df_ref[sens_feat].min())
    hi = float(df_ref[sens_feat].max())
else:
    lo, hi = RANGES[sens_feat]

grid_vals = np.linspace(lo, hi, 80)
df_sens = pd.DataFrame(
    {
        "deltaP_mmHg": deltaP,
        "gamma": gamma,
        "Q_ml_min": Q_perf,
        "Nterm": Nterm,
    },
    index=range(len(grid_vals)),
)
df_sens[sens_feat] = grid_vals
df_sens["pred_volume_mm3"] = mod_vol.predict(df_sens[features])
df_sens["pred_raio_root_mm"] = mod_raio.predict(df_sens[features])

valores_atuais = {
    "deltaP_mmHg": deltaP,
    "gamma": gamma,
    "Q_ml_min": Q_perf,
    "Nterm": Nterm,
}

sc1, sc2 = st.columns(2)
for coluna, alvo, titulo, cor in [
    (
        sc1,
        "pred_volume_mm3",
        f"{LABELS['volume_mm3']} previsto (mm³)",
        COLORS["primary"],
    ),
    (
        sc2,
        "pred_raio_root_mm",
        f"{LABELS['raio_root_mm']} previsto (mm)",
        COLORS["teal"],
    ),
]:
    with coluna:
        fig_sens = px.line(
            df_sens,
            x=sens_feat,
            y=alvo,
            labels={sens_feat: LABELS[sens_feat]},
            markers=False,
            title=titulo,
            color_discrete_sequence=[cor],
        )
        fig_sens.add_vline(
            x=valores_atuais[sens_feat],
            line_color=COLORS["ink"],
            line_dash="dot",
            annotation_text="atual",
            annotation_position="top",
        )
        fig_sens = plotly_theme(fig_sens, height=340, showlegend=False)
        st.plotly_chart(fig_sens, width="stretch", config=PLOTLY_CONFIG, theme=None)
