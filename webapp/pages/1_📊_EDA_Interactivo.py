import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ui import (
    COLORS,
    COLORWAY,
    CORR_SCALE,
    LABEL_HINT,
    LABELS,
    PLOTLY_CONFIG,
    SEQUENTIAL_SCALE,
    page_header,
    plotly_theme,
)

page_header(
    "EDA interativo",
    "Explore distribuições, correlações e a variabilidade estocástica das árvores CCO.",
    icon="query_stats",
)
st.caption(LABEL_HINT)


@st.cache_data
def carregar_dados():
    base_dir = Path(__file__).resolve().parents[2]
    candidatos = [
        base_dir / "dados" / "banco_CCO_XAI_final_500_arvores.csv",
        base_dir / "banco_CCO_XAI_final_500_arvores.csv",
    ]
    for caminho in candidatos:
        if caminho.exists():
            return pd.read_csv(caminho)
    st.error("Arquivo de dados não encontrado.", icon=":material/folder_off:")
    return pd.DataFrame()


df = carregar_dados()
if df.empty:
    st.stop()

features = ["deltaP_mmHg", "gamma", "Q_ml_min", "Nterm"]
targets = ["volume_mm3", "raio_root_mm"]
main_vars = features + targets
steps = {"deltaP_mmHg": 0.5, "gamma": 0.01, "Q_ml_min": 10.0, "Nterm": 1}
all_configs = sorted(df["id_configuracao"].unique())

with st.sidebar:
    st.markdown("### :material/filter_list: Filtros")
    sel_configs = st.multiselect(
        "Configurações",
        all_configs,
        default=[],
        format_func=lambda c: f"Configuração {c}",
    )

    df_filt = df.copy()
    if sel_configs:
        df_filt = df_filt[df_filt["id_configuracao"].isin(sel_configs)]

    for feat in features:
        inteiro = pd.api.types.is_integer_dtype(df[feat])
        min_v = int(df[feat].min()) if inteiro else float(df[feat].min())
        max_v = int(df[feat].max()) if inteiro else float(df[feat].max())
        r = st.slider(
            LABELS[feat],
            min_value=min_v,
            max_value=max_v,
            value=(min_v, max_v),
            step=steps[feat],
        )
        df_filt = df_filt[(df_filt[feat] >= r[0]) & (df_filt[feat] <= r[1])]

    st.caption(f"Amostras selecionadas: **{len(df_filt)}** de {len(df)}")
    st.download_button(
        "Baixar dados filtrados",
        data=df_filt.to_csv(index=False).encode("utf-8-sig"),
        file_name="cco_xai_filtrado.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )

if df_filt.empty:
    st.warning(
        "Nenhuma amostra corresponde aos filtros selecionados.",
        icon=":material/filter_alt_off:",
    )
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Resumo",
        "Distribuições",
        "Correlação",
        "Dispersão",
        "Configurações",
    ]
)

with tab1:
    st.markdown("#### Resumo estatístico das variáveis principais")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Amostras", len(df_filt))
    k2.metric("Configurações", df_filt["id_configuracao"].nunique())
    k3.metric(
        f"Média · {LABELS['volume_mm3']}",
        f"{df_filt['volume_mm3'].mean():.1f} mm³",
    )
    k4.metric(
        f"Média · {LABELS['raio_root_mm']}",
        f"{df_filt['raio_root_mm'].mean():.3f} mm",
    )

    desc = df_filt[main_vars].describe().T
    desc.index = [LABELS.get(i, i) for i in desc.index]
    desc.columns = [
        "n",
        "Média",
        "Desvio padrão",
        "Mínimo",
        "25%",
        "Mediana",
        "75%",
        "Máximo",
    ]
    st.dataframe(desc.style.format("{:.3f}"), width="stretch")

    nulos = int(df_filt[main_vars].isna().sum().sum())
    duplicatas = int(df_filt.duplicated().sum())
    st.caption(f"Valores ausentes: {nulos} · Linhas duplicadas: {duplicatas}")

with tab2:
    c1, c2 = st.columns([2, 1])
    var_escolhida = c1.selectbox(
        "Variável",
        main_vars,
        index=4,
        format_func=lambda v: LABELS.get(v, v),
    )
    nbins = c2.slider("Número de faixas", min_value=10, max_value=60, value=30)

    serie = df_filt[var_escolhida]
    fig_hist = px.histogram(
        df_filt,
        x=var_escolhida,
        nbins=nbins,
        labels={var_escolhida: LABELS.get(var_escolhida, var_escolhida)},
        title=f"Distribuição de {LABELS.get(var_escolhida, var_escolhida)}",
        color_discrete_sequence=[COLORS["primary"]],
    )
    fig_hist.update_traces(marker_line_color="white", marker_line_width=0.6, opacity=0.9)
    fig_hist.add_vline(
        x=float(serie.mean()),
        line_dash="dot",
        line_color=COLORS["ink"],
        annotation_text="média",
        annotation_position="top",
    )
    fig_hist = plotly_theme(fig_hist, height=430)
    st.plotly_chart(fig_hist, width="stretch", config=PLOTLY_CONFIG, theme=None)

    m1, m2, m3 = st.columns(3)
    m1.metric("Média", f"{serie.mean():.3f}")
    m2.metric("Mediana", f"{serie.median():.3f}")
    m3.metric("Desvio padrão", f"{serie.std():.3f}")

with tab3:
    st.markdown("#### Correlação linear de Pearson")
    corr = df_filt[main_vars].corr(method="pearson")
    corr.index = [LABELS.get(i, i) for i in corr.index]
    corr.columns = [LABELS.get(i, i) for i in corr.columns]

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=CORR_SCALE,
        zmin=-1,
        zmax=1,
        title="Entradas e alvos",
    )
    fig_corr.update_layout(coloraxis_colorbar=dict(title="r", thickness=12, len=0.7))
    fig_corr = plotly_theme(fig_corr, height=560)
    fig_corr.update_xaxes(tickangle=-30, showgrid=False)
    fig_corr.update_yaxes(showgrid=False)
    st.plotly_chart(fig_corr, width="stretch", config=PLOTLY_CONFIG, theme=None)

    with st.expander("Ver tabela de correlações"):
        st.dataframe(corr.style.format("{:.2f}"), width="stretch")

with tab4:
    st.markdown("#### Diagrama de dispersão")
    c1, c2, c3, c4 = st.columns(4)
    x_ax = c1.selectbox(
        "Eixo X",
        main_vars,
        index=0,
        format_func=lambda v: LABELS.get(v, v),
    )
    y_ax = c2.selectbox(
        "Eixo Y",
        main_vars,
        index=4,
        format_func=lambda v: LABELS.get(v, v),
    )
    c_col = c3.selectbox(
        "Cor",
        ["Nenhum"] + main_vars + ["id_configuracao"],
        index=0,
        format_func=lambda v: "Nenhum" if v == "Nenhum" else LABELS.get(v, v),
    )
    mostrar_ols = c4.checkbox("Reta de tendência", value=False)

    color_param = None if c_col == "Nenhum" else c_col
    axis_labels = {
        x_ax: LABELS.get(x_ax, x_ax),
        y_ax: LABELS.get(y_ax, y_ax),
    }
    if color_param:
        axis_labels[color_param] = LABELS.get(color_param, color_param)

    kwargs = {
        "x": x_ax,
        "y": y_ax,
        "hover_data": ["id_arvore", "id_configuracao", "seed"],
        "labels": axis_labels,
        "title": f"{LABELS.get(x_ax, x_ax)} × {LABELS.get(y_ax, y_ax)}",
        "opacity": 0.65,
    }
    if color_param:
        kwargs["color"] = color_param
        kwargs["color_continuous_scale"] = SEQUENTIAL_SCALE
    else:
        kwargs["color_discrete_sequence"] = [COLORS["primary"]]

    if mostrar_ols and (color_param is None or df_filt[color_param].nunique() <= 5):
        kwargs["trendline"] = "ols"
        kwargs["trendline_color_override"] = COLORS["ink"]

    fig_scat = px.scatter(df_filt, **kwargs)
    fig_scat = plotly_theme(fig_scat, height=520)
    st.plotly_chart(fig_scat, width="stretch", config=PLOTLY_CONFIG, theme=None)

with tab5:
    st.markdown("#### Variabilidade entre sementes da mesma configuração")
    st.caption(
        "Cada configuração possui cinco árvores geradas com sementes distintas. "
        "Os boxplots mostram a dispersão de cada grupo."
    )

    configs_default = [1, 25, 50, 75]
    if not all(c in all_configs for c in configs_default):
        configs_default = all_configs[:4]

    sel_configs_comp = st.multiselect(
        "Configurações para comparar",
        all_configs,
        default=configs_default,
        format_func=lambda c: f"Configuração {c}",
    )

    if not sel_configs_comp:
        st.info("Selecione ao menos uma configuração.", icon=":material/info:")
    else:
        df_comp = df[df["id_configuracao"].isin(sel_configs_comp)].copy()
        df_comp["Configuração"] = "Cfg " + df_comp["id_configuracao"].astype(str)

        col_v, col_r = st.columns(2)
        for coluna, alvo, titulo in [
            (
                col_v,
                "volume_mm3",
                f"{LABELS['volume_mm3']} — volume intravascular total (mm³)",
            ),
            (
                col_r,
                "raio_root_mm",
                f"{LABELS['raio_root_mm']} — raio proximal da raiz (mm)",
            ),
        ]:
            with coluna:
                fig_box = px.box(
                    df_comp,
                    x="Configuração",
                    y=alvo,
                    color="Configuração",
                    points="all",
                    labels={alvo: LABELS[alvo]},
                    color_discrete_sequence=COLORWAY,
                    title=titulo,
                )
                fig_box.update_traces(marker=dict(size=5, opacity=0.55))
                fig_box = plotly_theme(fig_box, height=420, showlegend=False)
                st.plotly_chart(fig_box, width="stretch", config=PLOTLY_CONFIG, theme=None)

        with st.expander("Resumo por configuração"):
            resumo = (
                df_comp.groupby("Configuração")[targets]
                .agg(["mean", "std"])
                .round(3)
            )
            resumo.columns = [
                f"{LABELS['volume_mm3']} — média (mm³)",
                f"{LABELS['volume_mm3']} — desvio (mm³)",
                f"{LABELS['raio_root_mm']} — média (mm)",
                f"{LABELS['raio_root_mm']} — desvio (mm)",
            ]
            st.dataframe(resumo, width="stretch")
