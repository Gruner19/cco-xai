# AGENTS.md

CCO-XAI final project (UFOP/DECOM): tabular ML on synthetic arterial trees, a Streamlit dashboard, and LaTeX report/slide templates. This is not a git repo and has no test, lint, formatter, or CI configs — do not invent those commands.

## Environment and commands

- Python >= 3.10. Install with `pip install -r requirements.txt` (core versions are pinned; `shap`/`lime` are listed but not used yet).
- Run all commands from the repo root. Scripts resolve paths from `scripts/` via `__file__`; `02_modelos.py` also `chdir`s into `scripts/`.
- Pipeline order (02 must run before 03):
  1. `python scripts/01_eda.py` → EDA figures/tables
  2. `python scripts/02_modelos.py` → `modelos/*.joblib` (required before 03)
  3. `python scripts/03_avaliacao.py` → metrics and evaluation plots
- Dashboard: `streamlit run webapp/app.py`
- LaTeX: build from the nested `template-*/template-*/` subdirs. The report has a `latexmkrc` (so `latexmk -pdf` works); the README documents `pdflatex`.
- No test suite: verify by rerunning the affected pipeline step and inspecting generated files.

## Domain constraints (do not break)

- Model inputs are exactly `["deltaP_mmHg", "gamma", "Q_ml_min", "Nterm"]`; targets are `volume_mm3` and `raio_root_mm`. Keep this order and these names — serialized joblib models depend on them.
- Use only those 4 columns as predictors. `id_configuracao` is for grouping only; never randomly split rows. All 5 seeds of a configuration must stay in the same split (`GroupShuffleSplit`, `GroupKFold`).
- Keep `random_state=42`; reproducibility is an explicit grading requirement.

## Repo-specific gotchas

- The official dataset is `dados/banco_CCO_XAI_final_500_arvores.csv`. The copy at the repo root is only a fallback used by the webapp EDA page; the scripts read `dados/`.
- `resultados/tabelas/metricas_modelos.csv` uses `treino`/`teste`, while `divisao_train_test.csv` uses `train`/`test`. Keep this distinction when filtering metrics.
- Dashboard navigation is defined in `webapp/app.py` with `st.navigation`. Page files keep emoji-prefixed filenames, but titles and Material icons come from `st.Page`; adding, moving, or removing a page requires updating `app.py`.
- Shared colors, labels, Plotly theme, and CSS live in `webapp/ui.py`; the global Streamlit theme is in `.streamlit/config.toml`. Use `width="stretch"` for dataframes/charts instead of the deprecated `use_container_width`.
- LaTeX templates read images from a `figuras/` folder inside each template directory, not from `resultados/figuras/`. Copy regenerated PNGs into both template `figuras/` folders before rebuilding.
- Part II XAI (SHAP, permutation importance, PDP/ICE/ALE, LIME, stability) is not implemented. `webapp/pages/3_🔍_XAI_Explorer.py` is only a placeholder.
- Assignment spec and deadlines are in `Projeto_Integrador_CCO_XAI_Enunciado-v2.md` (Part I due 16/11/2026, final 09/12/2026); `README.md` documents the intended commands.
