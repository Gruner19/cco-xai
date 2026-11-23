# -*- coding: utf-8 -*-
"""
03_avaliacao.py
===============
Script de avaliação e comparação dos modelos treinados para o projeto CCO-XAI.

Carrega os modelos pré-treinados e os splits de dados salvos pelo script
02_modelos.py, calcula métricas de desempenho (MAE, RMSE, R²), gera gráficos
de observado vs. predito, análise de resíduos e comparação de R² entre modelos.

Saídas:
    - ../resultados/tabelas/metricas_modelos.csv
    - ../resultados/figuras/obs_vs_pred_volume_mm3.png
    - ../resultados/figuras/obs_vs_pred_raio_root_mm.png
    - ../resultados/figuras/residuos_volume_mm3.png
    - ../resultados/figuras/residuos_raio_root_mm.png
    - ../resultados/figuras/comparacao_r2_modelos.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Configurações gerais
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELOS_DIR = os.path.join(BASE_DIR, "..", "modelos")
FIGURAS_DIR = os.path.join(BASE_DIR, "..", "resultados", "figuras")
TABELAS_DIR = os.path.join(BASE_DIR, "..", "resultados", "tabelas")

DPI = 150

# Nomes internos (usados nos nomes de arquivo) e nomes de exibição
MODEL_KEYS = ["dummy", "linear_regression", "decision_tree", "random_forest", "xgboost"]
MODEL_DISPLAY = {
    "dummy": "Dummy",
    "linear_regression": "Regressão Linear",
    "decision_tree": "Árvore de Decisão",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

TARGETS = ["volume_mm3", "raio_root_mm"]
TARGET_UNITS = {"volume_mm3": "mm³", "raio_root_mm": "mm"}
TARGET_DISPLAY = {"volume_mm3": "Volume (mm³)", "raio_root_mm": "Raio da raiz (mm)"}


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _load_data():
    """Carrega os splits de treino e teste salvos pelo 02_modelos.py."""
    print("[INFO] Carregando dados de treino e teste...")
    X_train = load(os.path.join(MODELOS_DIR, "X_train.joblib"))
    X_test = load(os.path.join(MODELOS_DIR, "X_test.joblib"))
    y_train = {
        "volume_mm3": load(os.path.join(MODELOS_DIR, "y_train_volume.joblib")),
        "raio_root_mm": load(os.path.join(MODELOS_DIR, "y_train_raio.joblib")),
    }
    y_test = {
        "volume_mm3": load(os.path.join(MODELOS_DIR, "y_test_volume.joblib")),
        "raio_root_mm": load(os.path.join(MODELOS_DIR, "y_test_raio.joblib")),
    }
    print(f"       X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def _load_models():
    """Carrega todos os modelos treinados (por alvo)."""
    print("[INFO] Carregando modelos treinados...")
    models = {}
    for target in TARGETS:
        models[target] = {}
        for key in MODEL_KEYS:
            path = os.path.join(MODELOS_DIR, f"{key}_{target}.joblib")
            models[target][key] = load(path)
            print(f"       Carregado: {key} → {target}")
    return models


def _compute_metrics(y_true, y_pred):
    """Retorna MAE, RMSE e R² para um par (observado, predito)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ---------------------------------------------------------------------------
# Etapa 3.1 – Métricas de desempenho
# ---------------------------------------------------------------------------

def compute_all_metrics(models, X_train, X_test, y_train, y_test):
    """
    Calcula MAE, RMSE e R² para cada modelo × alvo × conjunto (treino/teste).
    Retorna um DataFrame consolidado e o salva em CSV.
    """
    print("\n" + "=" * 70)
    print("ETAPA 3.1 – Cálculo de métricas de desempenho")
    print("=" * 70)

    rows = []
    for target in TARGETS:
        for key in MODEL_KEYS:
            model = models[target][key]
            for conjunto, X, y in [("treino", X_train, y_train[target]),
                                   ("teste", X_test, y_test[target])]:
                y_pred = model.predict(X)
                mae, rmse, r2 = _compute_metrics(y, y_pred)
                rows.append({
                    "target": target,
                    "modelo": MODEL_DISPLAY[key],
                    "conjunto": conjunto,
                    "MAE": round(mae, 6),
                    "RMSE": round(rmse, 6),
                    "R2": round(r2, 6),
                })

    df_metrics = pd.DataFrame(rows)

    # Salvar CSV
    os.makedirs(TABELAS_DIR, exist_ok=True)
    csv_path = os.path.join(TABELAS_DIR, "metricas_modelos.csv")
    df_metrics.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[SALVO] {csv_path}")

    # Imprimir tabelas formatadas no console
    for target in TARGETS:
        unit = TARGET_UNITS[target]
        print(f"\n--- Métricas para: {TARGET_DISPLAY[target]} ---")
        sub = df_metrics[df_metrics["target"] == target].copy()
        sub_display = sub[["modelo", "conjunto", "MAE", "RMSE", "R2"]].copy()
        sub_display = sub_display.rename(columns={
            "MAE": f"MAE ({unit})",
            "RMSE": f"RMSE ({unit})",
            "R2": "R²",
        })
        print(sub_display.to_string(index=False))

    return df_metrics


# ---------------------------------------------------------------------------
# Etapa 3.2 – Gráficos de observado vs. predito
# ---------------------------------------------------------------------------

def plot_obs_vs_pred(models, X_test, y_test):
    """
    Para cada alvo, cria uma figura com 5 subplots (1 por modelo):
    scatter de y_test vs. y_pred com linha identidade (y = x).
    """
    print("\n" + "=" * 70)
    print("ETAPA 3.2 – Gráficos de observado vs. predito")
    print("=" * 70)

    os.makedirs(FIGURAS_DIR, exist_ok=True)

    for target in TARGETS:
        unit = TARGET_UNITS[target]
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)
        fig.suptitle(
            f"Observado vs. Predito – {TARGET_DISPLAY[target]}",
            fontsize=14, fontweight="bold",
        )

        for ax, key in zip(axes, MODEL_KEYS):
            model = models[target][key]
            y_true = y_test[target]
            y_pred = model.predict(X_test if not hasattr(X_test, "values") else X_test)

            r2 = r2_score(y_true, y_pred)

            ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidths=0.3, s=30)

            # Linha identidade
            lims = [
                min(np.min(y_true), np.min(y_pred)),
                max(np.max(y_true), np.max(y_pred)),
            ]
            margin = (lims[1] - lims[0]) * 0.05
            lims = [lims[0] - margin, lims[1] + margin]
            ax.plot(lims, lims, "r--", linewidth=1.2, label="y = x")

            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_title(f"{MODEL_DISPLAY[key]}\nR² = {r2:.4f}", fontsize=10)
            ax.set_xlabel(f"Observado ({unit})")
            if ax == axes[0]:
                ax.set_ylabel(f"Predito ({unit})")
            ax.legend(loc="upper left", fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig_path = os.path.join(FIGURAS_DIR, f"obs_vs_pred_{target}.png")
        fig.savefig(fig_path, dpi=DPI)
        plt.close(fig)
        print(f"[SALVO] {fig_path}")


# ---------------------------------------------------------------------------
# Etapa 3.3 – Análise de resíduos
# ---------------------------------------------------------------------------

def plot_residuals(models, X_test, y_test):
    """
    Para cada alvo, cria uma figura com 2 linhas × 5 colunas:
        - Linha 1: resíduos vs. predito (scatter)
        - Linha 2: histograma dos resíduos
    """
    print("\n" + "=" * 70)
    print("ETAPA 3.3 – Análise de resíduos")
    print("=" * 70)

    os.makedirs(FIGURAS_DIR, exist_ok=True)

    for target in TARGETS:
        unit = TARGET_UNITS[target]
        fig, axes = plt.subplots(2, 5, figsize=(22, 8))
        fig.suptitle(
            f"Análise de Resíduos – {TARGET_DISPLAY[target]}",
            fontsize=14, fontweight="bold",
        )

        for col_idx, key in enumerate(MODEL_KEYS):
            model = models[target][key]
            y_true = y_test[target]
            y_pred = model.predict(X_test)
            residuals = y_true - y_pred

            # --- Linha 1: resíduos vs. predito ---
            ax_scatter = axes[0, col_idx]
            ax_scatter.scatter(y_pred, residuals, alpha=0.5, edgecolors="k",
                               linewidths=0.3, s=30)
            ax_scatter.axhline(0, color="r", linestyle="--", linewidth=1.0)
            ax_scatter.set_title(MODEL_DISPLAY[key], fontsize=10)
            ax_scatter.set_xlabel(f"Predito ({unit})")
            if col_idx == 0:
                ax_scatter.set_ylabel(f"Resíduo ({unit})")

            # --- Linha 2: histograma dos resíduos ---
            ax_hist = axes[1, col_idx]
            ax_hist.hist(residuals, bins=25, edgecolor="black", alpha=0.7)
            ax_hist.axvline(0, color="r", linestyle="--", linewidth=1.0)
            ax_hist.set_xlabel(f"Resíduo ({unit})")
            if col_idx == 0:
                ax_hist.set_ylabel("Frequência")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig_path = os.path.join(FIGURAS_DIR, f"residuos_{target}.png")
        fig.savefig(fig_path, dpi=DPI)
        plt.close(fig)
        print(f"[SALVO] {fig_path}")


# ---------------------------------------------------------------------------
# Etapa 3.4 – Gráfico comparativo de R²
# ---------------------------------------------------------------------------

def plot_r2_comparison(df_metrics):
    """
    Gráfico de barras comparando R² (no conjunto de teste) dos modelos,
    lado a lado para ambos os alvos.
    """
    print("\n" + "=" * 70)
    print("ETAPA 3.4 – Comparação de R² entre modelos")
    print("=" * 70)

    os.makedirs(FIGURAS_DIR, exist_ok=True)

    # Filtrar apenas conjunto de teste
    df_test = df_metrics[df_metrics["conjunto"] == "teste"].copy()
    df_test["target_display"] = df_test["target"].map(TARGET_DISPLAY)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Posições das barras
    n_models = len(MODEL_KEYS)
    n_targets = len(TARGETS)
    x = np.arange(n_models)
    width = 0.35

    colors = sns.color_palette("Set2", n_targets)

    for i, target in enumerate(TARGETS):
        subset = df_test[df_test["target"] == target]
        # Garantir ordem dos modelos
        r2_values = []
        for key in MODEL_KEYS:
            display_name = MODEL_DISPLAY[key]
            val = subset.loc[subset["modelo"] == display_name, "R2"].values[0]
            r2_values.append(val)

        offset = (i - (n_targets - 1) / 2) * width
        bars = ax.bar(x + offset, r2_values, width, label=TARGET_DISPLAY[target],
                      color=colors[i], edgecolor="black", linewidth=0.5)

        # Rótulos nas barras
        for bar, val in zip(bars, r2_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[k] for k in MODEL_KEYS], fontsize=9)
    ax.set_ylabel("R² (conjunto de teste)", fontsize=11)
    ax.set_title("Comparação de R² entre Modelos", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0] - 0.05))

    fig.tight_layout()
    fig_path = os.path.join(FIGURAS_DIR, "comparacao_r2_modelos.png")
    fig.savefig(fig_path, dpi=DPI)
    plt.close(fig)
    print(f"[SALVO] {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Função principal de avaliação dos modelos."""
    print("=" * 70)
    print("  SCRIPT 03 – Avaliação e Comparação de Modelos (CCO-XAI)")
    print("=" * 70)

    # Carregar dados e modelos
    X_train, X_test, y_train, y_test = _load_data()
    models = _load_models()

    # Etapa 3.1 – Métricas
    df_metrics = compute_all_metrics(models, X_train, X_test, y_train, y_test)

    # Etapa 3.2 – Observado vs. Predito
    plot_obs_vs_pred(models, X_test, y_test)

    # Etapa 3.3 – Resíduos
    plot_residuals(models, X_test, y_test)

    # Etapa 3.4 – Comparação de R²
    plot_r2_comparison(df_metrics)

    print("\n" + "=" * 70)
    print("  Avaliação concluída com sucesso!")
    print("=" * 70)


if __name__ == "__main__":
    main()
