import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ui import (
    COLORS,
    DIVERGING_SCALE,
    LABEL_HINT,
    PLOTLY_CONFIG,
    TARGET_LABELS,
    TARGET_UNITS,
    page_header,
    plotly_theme,
)

page_header(
    "Comparador",
    "Métricas de generalização, observado vs. previsto e análise de resíduos.",
    icon="leaderboard",
)
st.caption(LABEL_HINT)

base_dir = Path(__file__).resolve().parents[2]
tab_dir = base_dir / "resultados" / "tabelas"
models_dir = base_dir / "modelos"
metricas_csv = tab_dir / "metricas_modelos.csv"

TARGETS = ["volume_mm3", "raio_root_mm"]
MODELOS = {
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Árvore de Decisão": "decision_tree",
    "Regressão Linear": "linear_regression",
    "Dummy (Média)": "dummy",
}


@st.cache_data
def carregar_metricas():
    if metricas_csv.exists():
        return pd.read_csv(metricas_csv)
    return None


df_met = carregar_metricas()
if df_met is None or df_met.empty:
    st.warning(
        "A tabela de métricas ainda não foi gerada. Execute "
        "`python scripts/03_avaliacao.py` e recarregue a página.",
        icon=":material/hourglass_empty:",
    )
    st.stop()

with st.sidebar:
    st.markdown("### :material/tune: Avaliação")
    target_sel = st.selectbox(
        "Alvo",
        TARGETS,
        format_func=lambda t: TARGET_LABELS[t],
    )
    modelo_nome = st.selectbox("Modelo", list(MODELOS.keys()), index=0)
    st.caption(
        "O alvo e o modelo selecionados controlam a tabela de métricas e as "
        "análises de observado vs. previsto e de resíduos."
    )

modelo_key = MODELOS[modelo_nome]
unit = TARGET_UNITS[target_sel]


def carregar_avaliacao(target: str, model_key: str):
    x_path = models_dir / "X_test.joblib"
    y_path = models_dir / f"y_test_{target.split('_')[0]}.joblib"
    m_path = models_dir / f"{model_key}_{target}.joblib"

    if not (x_path.exists() and y_path.exists() and m_path.exists()):
        return None
    return joblib.load(x_path), joblib.load(y_path), joblib.load(m_path)


avaliacao = carregar_avaliacao(target_sel, modelo_key)
df_plot = None
mae = rmse = r2 = None

if avaliacao is not None:
    X_test, y_test, model = avaliacao
    y_pred = model.predict(X_test)
    df_plot = pd.DataFrame(
        {
            "Observado": y_test,
            "Previsto": y_pred,
            "Resíduo": y_test - y_pred,
        }
    )
    residuo = df_plot["Resíduo"]
    mae = float(residuo.abs().mean())
    rmse = float((residuo**2).mean() ** 0.5)
    ss_res = float((residuo**2).sum())
    ss_tot = float(((df_plot["Observado"] - df_plot["Observado"].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Métricas",
        "Observado vs. previsto",
        "Resíduos",
        "Ranking",
    ]
)

with tab1:
    st.markdown(f"#### Desempenho · {TARGET_LABELS[target_sel]}")
    split_choice = st.radio(
        "Conjunto de avaliação",
        ["Todos", "teste", "treino"],
        horizontal=True,
    )

    df_sub = df_met[df_met["target"] == target_sel].copy()
    if split_choice != "Todos":
        df_sub = df_sub[df_sub["conjunto"] == split_choice]

    df_show = df_sub[["modelo", "conjunto", "MAE", "RMSE", "R2"]].rename(
        columns={
            "modelo": "Modelo",
            "conjunto": "Conjunto",
            "R2": "R²",
        }
    )

    st.caption(f"Unidade de erro: {unit} · Alvo selecionado na barra lateral.")
    st.dataframe(
        df_show.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Baixar tabela de métricas",
        data=df_met.to_csv(index=False).encode("utf-8-sig"),
        file_name="metricas_modelos.csv",
        mime="text/csv",
        icon=":material/download:",
    )

with tab2:
    st.markdown(f"#### {modelo_nome} · {TARGET_LABELS[target_sel]}")

    if df_plot is None:
        st.warning(
            "Os splits ou o modelo selecionado não foram encontrados. "
            "Execute `python scripts/02_modelos.py`.",
            icon=":material/hourglass_empty:",
        )
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{mae:.3f} {unit}")
        m2.metric("RMSE", f"{rmse:.3f} {unit}")
        m3.metric("R²", f"{r2:.4f}")

        lim_lo = float(min(df_plot["Observado"].min(), df_plot["Previsto"].min()))
        lim_hi = float(max(df_plot["Observado"].max(), df_plot["Previsto"].max()))
        pad = (lim_hi - lim_lo) * 0.04 if lim_hi > lim_lo else 1.0
        lims = [lim_lo - pad, lim_hi + pad]

        fig_obs = px.scatter(
            df_plot,
            x="Observado",
            y="Previsto",
            color="Resíduo",
            color_continuous_scale=DIVERGING_SCALE,
            color_continuous_midpoint=0,
            hover_data=["Resíduo"],
            labels={
                "Observado": f"Observado — {TARGET_LABELS[target_sel]} · {unit}",
                "Previsto": f"Previsto — {TARGET_LABELS[target_sel]} · {unit}",
            },
            title="Observado vs. previsto no conjunto de teste",
        )
        fig_obs.update_traces(
            marker=dict(size=7, line=dict(width=0.5, color="white")),
            opacity=0.85,
        )
        fig_obs.add_shape(
            type="line",
            x0=lims[0],
            y0=lims[0],
            x1=lims[1],
            y1=lims[1],
            line=dict(color=COLORS["ink"], dash="dash", width=1),
        )
        fig_obs.update_xaxes(range=lims)
        fig_obs.update_yaxes(range=lims)
        fig_obs.update_layout(
            coloraxis_colorbar=dict(title="Resíduo", thickness=12, len=0.7)
        )
        fig_obs = plotly_theme(fig_obs, height=560, showlegend=False)
        st.plotly_chart(fig_obs, width="stretch", config=PLOTLY_CONFIG, theme=None)

with tab3:
    st.markdown(f"#### Análise de resíduos · {modelo_nome} · {TARGET_LABELS[target_sel]}")

    if df_plot is None:
        st.warning(
            "Os splits ou o modelo selecionado não foram encontrados. "
            "Execute `python scripts/02_modelos.py`.",
            icon=":material/hourglass_empty:",
        )
    else:
        r1, r2, r3 = st.columns(3)
        r1.metric("Resíduo médio", f"{df_plot['Resíduo'].mean():.3f} {unit}")
        r2.metric("Desvio dos resíduos", f"{df_plot['Resíduo'].std():.3f} {unit}")
        r3.metric("Erro absoluto máximo", f"{df_plot['Resíduo'].abs().max():.3f} {unit}")

        c1, c2 = st.columns(2)
        with c1:
            fig_res = px.scatter(
                df_plot,
                x="Previsto",
                y="Resíduo",
                opacity=0.7,
                color_discrete_sequence=[COLORS["primary"]],
                labels={
                    "Previsto": f"Previsto — {TARGET_LABELS[target_sel]} · {unit}",
                    "Resíduo": f"Resíduo — {TARGET_LABELS[target_sel]} · {unit}",
                },
                title="Resíduos vs. previsto",
            )
            fig_res.add_hline(
                y=0,
                line_dash="dash",
                line_color=COLORS["ink"],
                line_width=1,
            )
            fig_res = plotly_theme(fig_res, height=360, showlegend=False)
            st.plotly_chart(fig_res, width="stretch", config=PLOTLY_CONFIG, theme=None)

        with c2:
            fig_hist = px.histogram(
                df_plot,
                x="Resíduo",
                nbins=24,
                color_discrete_sequence=[COLORS["teal"]],
                labels={
                    "Resíduo": f"Resíduo — {TARGET_LABELS[target_sel]} · {unit}"
                },
                title="Distribuição dos resíduos",
            )
            fig_hist.update_traces(
                marker_line_color="white",
                marker_line_width=0.5,
                opacity=0.9,
            )
            fig_hist.add_vline(
                x=0,
                line_dash="dash",
                line_color=COLORS["ink"],
                line_width=1,
            )
            fig_hist = plotly_theme(fig_hist, height=360, showlegend=False)
            st.plotly_chart(fig_hist, width="stretch", config=PLOTLY_CONFIG, theme=None)

with tab4:
    st.markdown("#### Ranking de R² no conjunto de teste")
    df_test = df_met[df_met["conjunto"] == "teste"].copy()

    if df_test.empty:
        st.info("Não há resultados de teste na tabela de métricas.", icon=":material/info:")
    else:
        df_test["Alvo"] = df_test["target"].map(TARGET_LABELS)
        fig_bar = px.bar(
            df_test,
            x="modelo",
            y="R2",
            color="Alvo",
            barmode="group",
            text_auto=".3f",
            color_discrete_map={
                TARGET_LABELS["volume_mm3"]: COLORS["primary"],
                TARGET_LABELS["raio_root_mm"]: COLORS["ink"],
            },
            category_orders={"modelo": list(MODELOS.keys())},
            labels={"modelo": "Modelo", "R2": "R²"},
            title="R² por modelo e alvo",
        )
        fig_bar.add_hline(y=0, line_color=COLORS["border"], line_width=1)
        fig_bar = plotly_theme(fig_bar, height=480)
        st.plotly_chart(fig_bar, width="stretch", config=PLOTLY_CONFIG, theme=None)

        with st.expander("Ver ranking em tabela"):
            ranking = (
                df_test[["Alvo", "modelo", "R2"]]
                .rename(columns={"modelo": "Modelo", "R2": "R²"})
                .sort_values(["Alvo", "R²"], ascending=[True, False])
            )
            st.dataframe(
                ranking.style.format({"R²": "{:.4f}"}),
                hide_index=True,
                width="stretch",
            )
