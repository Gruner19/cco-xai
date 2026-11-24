# CCO-XAI Project Instructions

This project focuses on tabular ML for synthetic arterial trees with a Streamlit dashboard and LaTeX documentation.

## Pipeline and Environment
- **Python:** >= 3.10. Install core dependencies with `pip install -r requirements.txt`.
- **Root Execution:** Run all commands from the repo root. Scripts in `scripts/` resolve paths via `__file__`.
- **Pipeline Order:** Strict dependency order:
  1. `python scripts/01_eda.py` (generates EDA figures/tables)
  2. `python scripts/02_modelos.py` (generates `modelos/*.joblib`, `chdir`s into `scripts/`)
  3. `python scripts/03_avaliacao.py` (requires step 2; generates evaluation metrics/plots)
- **Dashboard:** Run with `streamlit run webapp/app.py`.
- **LaTeX:** Build nested templates via `latexmk -pdf` (uses `latexmkrc`).

## Domain Constraints (Critical)
- **Predictors:** Use exactly `["deltaP_mmHg", "gamma", "Q_ml_min", "Nterm"]`.
- **Targets:** `volume_mm3` and `raio_root_mm`.
- **Data Integrity:** Never randomly split rows. All 5 seeds of a configuration must stay together (`GroupShuffleSplit` or `GroupKFold`). Use `random_state=42` for reproducibility.
- **Data Source:** Primary data is in `dados/banco_CCO_XAI_final_500_arvores.csv`. The copy at root is for EDA only.

## Repository Structure Quirks
- **Metrics:** `resultados/tabelas/metricas_modelos.csv` uses `treino`/`teste`, while `divisao_train_test.csv` uses `train`/`test`.
- **Dashboard UI:** Navigation defined in `webapp/app.py`. Shared styles in `webapp/ui.py`. Use `width="stretch"` for dataframes/charts.
- **LaTeX:** Templates use local `figuras/` folders. Regenerated PNGs must be copied into each template's `figuras/` folder.
- **Placeholder:** `webapp/pages/3_🔍_XAI_Explorer.py` is currently unimplemented.
