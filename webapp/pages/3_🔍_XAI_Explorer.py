import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ui import LABEL_HINT, page_header

page_header(
    "XAI Explorer",
    "Explicabilidade global, local e estabilidade das explicações.",
    icon="hub",
)
st.caption(LABEL_HINT)

st.info(
    "Módulo planejado para a Parte II (entrega em 09/12/2026). O foco será "
    "o melhor modelo de ensemble para volume_mm3 (V_intra), mantendo "
    "raio_root_mm (r_root) na comparação preditiva e na importância global.",
    icon=":material/schedule:",
)

c1, c2 = st.columns(2)

with c1:
    with st.container(border=True):
        st.markdown("#### :material/public: Explicabilidade global")
        st.markdown(
            "- :material/check_box_outline_blank: **SHAP** — importância média "
            "absoluta, beeswarm e dependência.\n"
            "- :material/check_box_outline_blank: **Importância por permutação** — "
            "perda de desempenho no conjunto de teste.\n"
            "- :material/check_box_outline_blank: **PDP, ICE ou ALE** — efeito "
            "marginal de uma entrada relevante.\n"
            "- :material/check_box_outline_blank: **Comparação de rankings** — "
            "SHAP vs. permutação."
        )

with c2:
    with st.container(border=True):
        st.markdown("#### :material/center_focus_strong: Explicabilidade local")
        st.markdown(
            "- :material/check_box_outline_blank: **Seleção de casos** — duas "
            "árvores de teste, uma de volume_mm3 (V_intra) baixo e outra alto.\n"
            "- :material/check_box_outline_blank: **SHAP waterfall** — "
            "decomposição local da predição.\n"
            "- :material/check_box_outline_blank: **LIME tabular** — "
            "aproximação linear local.\n"
            "- :material/check_box_outline_blank: **Estabilidade** — repetição "
            "do ranking em três divisões agrupadas."
        )

st.markdown("---")
st.caption(
    "Referência: Kamath & Liu (2021). As explicações post-hoc descrevem o "
    "comportamento do modelo, não relações causais ou mecanismos físicos do CCO."
)
