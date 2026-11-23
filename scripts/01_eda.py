#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Análise Exploratória de Dados (EDA) completa para o projeto CCO-XAI.

Carrega o banco de dados de árvores vasculares geradas pelo algoritmo CCO,
calcula estatísticas descritivas, gera histogramas, pairplots, mapa de
correlação, dispersões features × targets e comparação entre configurações.

Autores: Projeto CCO-XAI — UFOP
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # Backend não-interativo

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ── Caminhos relativos a partir do diretório scripts/ ────────────────────────
CAMINHO_CSV = os.path.join(os.path.dirname(__file__), "..", "dados",
                           "banco_CCO_XAI_final_500_arvores.csv")
DIR_FIGURAS = os.path.join(os.path.dirname(__file__), "..", "resultados", "figuras")
DIR_TABELAS = os.path.join(os.path.dirname(__file__), "..", "resultados", "tabelas")

# Variáveis de interesse
FEATURES = ["deltaP_mmHg", "gamma", "Q_ml_min", "Nterm"]
TARGETS  = ["volume_mm3", "raio_root_mm"]
VARIAVEIS_PRINCIPAIS = FEATURES + TARGETS

# Configuração visual
DPI = 150
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 10,
})


# ═════════════════════════════════════════════════════════════════════════════
# Funções auxiliares
# ═════════════════════════════════════════════════════════════════════════════

def carregar_dados(caminho: str) -> pd.DataFrame:
    """Carrega o CSV e imprime informações básicas."""
    print("─" * 60)
    print(f"Carregando dados de: {os.path.abspath(caminho)}")
    df = pd.read_csv(caminho)
    print(f"Shape: {df.shape}  |  Colunas: {df.shape[1]}")
    print(f"Nulos totais: {df.isnull().sum().sum()}")
    print(f"Duplicatas (linhas inteiras): {df.duplicated().sum()}")
    return df


def estatisticas_descritivas(df: pd.DataFrame, caminho_saida: str) -> None:
    """Etapa 1.2 — Calcula e salva estatísticas descritivas das 6 variáveis."""
    print("\n─── Estatísticas Descritivas ───")
    sub = df[VARIAVEIS_PRINCIPAIS]

    # dtypes
    print("\nTipos de dados:")
    print(sub.dtypes.to_string())

    # describe
    desc = sub.describe().T
    desc["faixa"] = desc["max"] - desc["min"]
    print("\n", desc.to_string())

    # Consistência de faixas
    print("\nVerificação de faixas (min, max):")
    for col in VARIAVEIS_PRINCIPAIS:
        print(f"  {col:20s}  min={df[col].min():.6g}  max={df[col].max():.6g}")

    # Salvar tabela
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    desc.to_csv(caminho_saida, encoding="utf-8-sig")
    print(f"\nTabela salva em: {os.path.abspath(caminho_saida)}")


def histogramas(df: pd.DataFrame, caminho_saida: str) -> None:
    """Etapa 1.3 — Grade 2×3 de histogramas com KDE."""
    print("\n─── Histogramas ───")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    rotulos = {
        "deltaP_mmHg":  "ΔP (mmHg)",
        "gamma":        "γ (expoente de bifurcação)",
        "Q_ml_min":     "Q (mL/min)",
        "Nterm":        "Nterm (terminais)",
        "volume_mm3":   "Volume (mm³)",
        "raio_root_mm": "Raio raiz (mm)",
    }

    for ax, var in zip(axes, VARIAVEIS_PRINCIPAIS):
        sns.histplot(df[var], kde=True, ax=ax, color="steelblue", edgecolor="white")
        ax.set_xlabel(rotulos.get(var, var))
        ax.set_ylabel("Frequência")
        ax.set_title(f"Distribuição de {rotulos.get(var, var)}")

    fig.suptitle("Histogramas das Variáveis Principais", fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    fig.savefig(caminho_saida, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {os.path.abspath(caminho_saida)}")


def pairplot_features(df: pd.DataFrame, caminho_saida: str) -> None:
    """Etapa 1.3 — Pairplot das 4 features, colorido pelo volume discretizado."""
    print("\n─── Pairplot das Features ───")
    tmp = df[FEATURES].copy()
    tmp["Volume (faixa)"] = pd.cut(
        df["volume_mm3"],
        bins=3,
        labels=["baixo", "médio", "alto"],
    )

    g = sns.pairplot(
        tmp,
        hue="Volume (faixa)",
        palette="viridis",
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 25},
    )
    g.figure.suptitle(
        "Pairplot das Features (cor = faixa de volume)", y=1.02, fontsize=13
    )
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    g.savefig(caminho_saida, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Figura salva em: {os.path.abspath(caminho_saida)}")


def correlacao_heatmap(df: pd.DataFrame, caminho_saida: str) -> None:
    """Etapa 1.3 — Mapa de calor da correlação de Pearson (6 variáveis)."""
    print("\n─── Correlação de Pearson ───")
    corr = df[VARIAVEIS_PRINCIPAIS].corr(method="pearson")
    print(corr.to_string())

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        ax=ax,
    )
    ax.set_title("Matriz de Correlação de Pearson", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    fig.savefig(caminho_saida, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {os.path.abspath(caminho_saida)}")


def dispersao_features_targets(df: pd.DataFrame, caminho_saida: str) -> None:
    """Etapa 1.3 — Grade 4×2 de dispersão (cada feature × cada target)."""
    print("\n─── Dispersão Features × Targets ───")
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))

    rotulos_feat = {
        "deltaP_mmHg": "ΔP (mmHg)",
        "gamma":       "γ",
        "Q_ml_min":    "Q (mL/min)",
        "Nterm":       "Nterm",
    }
    rotulos_tgt = {
        "volume_mm3":   "Volume (mm³)",
        "raio_root_mm": "Raio raiz (mm)",
    }

    for i, feat in enumerate(FEATURES):
        for j, tgt in enumerate(TARGETS):
            ax = axes[i, j]
            ax.scatter(df[feat], df[tgt], alpha=0.5, s=18, color="teal")
            ax.set_xlabel(rotulos_feat.get(feat, feat))
            ax.set_ylabel(rotulos_tgt.get(tgt, tgt))
            ax.set_title(f"{rotulos_feat.get(feat, feat)}  vs  {rotulos_tgt.get(tgt, tgt)}")

    fig.suptitle("Dispersão: Features × Targets", fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    fig.savefig(caminho_saida, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {os.path.abspath(caminho_saida)}")


def dispersao_compacta(df: pd.DataFrame, caminho_saida: str) -> None:
    """Etapa 1.3 — Grade 2×2 com dispersões selecionadas features × targets."""
    print("\n─── Dispersão Compacta Features × Targets ───")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    rotulos_feat = {
        "deltaP_mmHg": "ΔP (mmHg)",
        "gamma":       "γ",
        "Q_ml_min":    "Q (mL/min)",
        "Nterm":       "Nterm",
    }
    rotulos_tgt = {
        "volume_mm3":   "Volume (mm³)",
        "raio_root_mm": "Raio raiz (mm)",
    }
    paineis = [
        ("Q_ml_min", "volume_mm3"),
        ("Nterm", "volume_mm3"),
        ("gamma", "raio_root_mm"),
        ("deltaP_mmHg", "volume_mm3"),
    ]

    for ax, (feat, tgt) in zip(axes.flat, paineis):
        ax.scatter(df[feat], df[tgt], alpha=0.5, s=18, color="teal")
        ax.set_xlabel(rotulos_feat.get(feat, feat))
        ax.set_ylabel(rotulos_tgt.get(tgt, tgt))
        ax.set_title(f"{rotulos_feat.get(feat, feat)}  vs  {rotulos_tgt.get(tgt, tgt)}")

    fig.suptitle("Dispersões selecionadas: entradas × alvos", fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    fig.savefig(caminho_saida, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {os.path.abspath(caminho_saida)}")


def comparacao_configuracoes(df: pd.DataFrame, caminho_saida: str) -> None:
    """Etapa 1.4 — Boxplot/violin das 4 configurações selecionadas (ids 1, 25, 50, 75)."""
    print("\n─── Comparação entre Configurações ───")
    ids_selecionados = [1, 25, 50, 75]
    sub = df[df["id_configuracao"].isin(ids_selecionados)].copy()
    sub["id_configuracao"] = sub["id_configuracao"].astype(str)
    print(f"Configurações selecionadas: {ids_selecionados}  →  {len(sub)} linhas")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Volume
    sns.violinplot(
        data=sub, x="id_configuracao", y="volume_mm3",
        hue="id_configuracao", legend=False,
        inner="box", palette="Set2", ax=axes[0],
    )
    axes[0].set_xlabel("Configuração (id)")
    axes[0].set_ylabel("Volume (mm³)")
    axes[0].set_title("Volume por Configuração")

    # Raio raiz
    sns.violinplot(
        data=sub, x="id_configuracao", y="raio_root_mm",
        hue="id_configuracao", legend=False,
        inner="box", palette="Set2", ax=axes[1],
    )
    axes[1].set_xlabel("Configuração (id)")
    axes[1].set_ylabel("Raio raiz (mm)")
    axes[1].set_title("Raio Raiz por Configuração")

    fig.suptitle("Comparação entre Configurações Selecionadas", fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    fig.savefig(caminho_saida, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {os.path.abspath(caminho_saida)}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Executa todas as etapas da Análise Exploratória de Dados."""
    print("=" * 60)
    print("  ANÁLISE EXPLORATÓRIA DE DADOS — PROJETO CCO-XAI")
    print("=" * 60)

    # 1. Carga
    df = carregar_dados(CAMINHO_CSV)

    # 2. Estatísticas descritivas
    estatisticas_descritivas(
        df, os.path.join(DIR_TABELAS, "estatisticas_descritivas.csv")
    )

    # 3. Histogramas
    histogramas(
        df, os.path.join(DIR_FIGURAS, "histogramas_variaveis.png")
    )

    # 4. Pairplot
    pairplot_features(
        df, os.path.join(DIR_FIGURAS, "pairplot_features.png")
    )

    # 5. Correlação
    correlacao_heatmap(
        df, os.path.join(DIR_FIGURAS, "correlacao_heatmap.png")
    )

    # 6. Dispersão features × targets
    dispersao_features_targets(
        df, os.path.join(DIR_FIGURAS, "dispersao_features_targets.png")
    )

    # 7. Dispersão compacta para o relatório
    dispersao_compacta(
        df, os.path.join(DIR_FIGURAS, "dispersao_compacta.png")
    )

    # 8. Comparação de configurações
    comparacao_configuracoes(
        df, os.path.join(DIR_FIGURAS, "comparacao_configuracoes.png")
    )

    print("\n" + "=" * 60)
    print("  EDA concluída com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
