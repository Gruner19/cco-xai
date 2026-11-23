from pathlib import Path
import sys

import pandas as pd
import streamlit as st

WEBAPP_DIR = Path(__file__).resolve().parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

from ui import LABEL_HINT, LABELS, apply_global_style, page_header

st.set_page_config(
    page_title="CCO-XAI | UFOP",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_style()


def render_home() -> None:
    page_header(
        "Plataforma CCO-XAI",
        "Modelagem preditiva e explicabilidade em árvores arteriais sintéticas.",
        icon="hub",
    )

    st.markdown(
        "Painel analítico do Projeto Integrador da disciplina de Inteligência "
        "Artificial Explicável. Reúne exploração de dados, previsões em tempo "
        "real e avaliação dos modelos treinados sobre 500 árvores CCO."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### :material/dataset: Dados")
            st.markdown(
                "- **500** árvores sintéticas\n"
                "- **100** configurações\n"
                "- **5** sementes por configuração"
            )
    with c2:
        with st.container(border=True):
            st.markdown("#### :material/model_training: Modelos")
            st.markdown(
                "- Dummy e Regressão Linear\n"
                "- Árvore de Decisão\n"
                "- Random Forest e XGBoost"
            )
    with c3:
        with st.container(border=True):
            st.markdown("#### :material/rule: Avaliação")
            st.markdown(
                "- MAE, RMSE e R²\n"
                "- Validação agrupada por configuração\n"
                "- Teste final reservado (~20%)"
            )

    st.markdown("#### :material/table_chart: Variáveis do estudo")
    df_variaveis = pd.DataFrame(
        [
            {
                "Papel": "Entrada",
                "Variável": LABELS["deltaP_mmHg"],
                "Descrição": "Queda de pressão (ΔP)",
                "Unidade": "mmHg",
            },
            {
                "Papel": "Entrada",
                "Variável": LABELS["gamma"],
                "Descrição": "Expoente de bifurcação (γ)",
                "Unidade": "adimensional",
            },
            {
                "Papel": "Entrada",
                "Variável": LABELS["Q_ml_min"],
                "Descrição": "Vazão total de perfusão",
                "Unidade": "mL/min",
            },
            {
                "Papel": "Entrada",
                "Variável": LABELS["Nterm"],
                "Descrição": "Número de segmentos terminais",
                "Unidade": "contagem",
            },
            {
                "Papel": "Alvo",
                "Variável": LABELS["volume_mm3"],
                "Descrição": "Volume intravascular total",
                "Unidade": "mm³",
            },
            {
                "Papel": "Alvo",
                "Variável": LABELS["raio_root_mm"],
                "Descrição": "Raio do segmento proximal raiz",
                "Unidade": "mm",
            },
        ]
    )
    st.dataframe(df_variaveis, hide_index=True, width="stretch")
    st.caption(LABEL_HINT)

    st.markdown("#### :material/explore: Como navegar")
    st.markdown(
        "- :material/query_stats: **EDA interativo** — estatísticas, distribuições, "
        "correlações e variabilidade entre sementes.\n"
        "- :material/tune: **Predictor** — previsões e curvas de sensibilidade "
        "para os quatro parâmetros de entrada.\n"
        "- :material/leaderboard: **Comparador** — métricas de treino/teste, "
        "observado vs. previsto e resíduos.\n"
        "- :material/hub: **XAI Explorer** — análises globais e locais da Parte II."
    )

    st.info(
        "A Parte I está implementada. Os módulos de XAI (SHAP, LIME, importância "
        "por permutação e estabilidade) serão incorporados na entrega final.",
        icon=":material/schedule:",
    )


paginas = {
    "Início": [
        st.Page(
            render_home,
            title="Visão geral",
            icon=":material/space_dashboard:",
            url_path="visao-geral",
            default=True,
        ),
    ],
    "Análise": [
        st.Page(
            "pages/1_📊_EDA_Interactivo.py",
            title="EDA interativo",
            icon=":material/query_stats:",
            url_path="eda",
        ),
    ],
    "Modelagem": [
        st.Page(
            "pages/2_🤖_Predictor.py",
            title="Predictor",
            icon=":material/tune:",
            url_path="predictor",
        ),
        st.Page(
            "pages/4_📈_Comparador.py",
            title="Comparador",
            icon=":material/leaderboard:",
            url_path="comparador",
        ),
    ],
    "Explicabilidade": [
        st.Page(
            "pages/3_🔍_XAI_Explorer.py",
            title="XAI Explorer",
            icon=":material/hub:",
            url_path="xai",
        ),
    ],
}

with st.sidebar:
    st.markdown(
        '<div class="cco-brand">CCO<span>·</span>XAI</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cco-tagline">Plataforma CCO/UFOP</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

pg = st.navigation(paginas, expanded=True)

with st.sidebar:
    st.markdown("---")
    st.caption("Projeto Integrador · UFOP/DECOM · 2026")

pg.run()
