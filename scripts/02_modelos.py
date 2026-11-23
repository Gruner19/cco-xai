#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_modelos.py — Construção, ajuste e serialização de modelos de ML
===================================================================
Projeto CCO-XAI  •  Universidade Federal de Ouro Preto (UFOP)

Este script:
  1. Carrega o banco de dados pré-processado.
  2. Realiza a divisão treino/teste respeitando agrupamentos (id_configuracao).
  3. Treina 5 regressores para cada variável-alvo (volume_mm3, raio_root_mm):
     Dummy, Regressão Linear, Árvore de Decisão, Random Forest e XGBoost.
  4. Salva modelos treinados, hiperparâmetros e dados de divisão.
"""

# ── Importações ──────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from xgboost import XGBRegressor
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# ── Constantes ───────────────────────────────────────────────────────────────
FEATURES = ["deltaP_mmHg", "gamma", "Q_ml_min", "Nterm"]
TARGETS = ["volume_mm3", "raio_root_mm"]
GROUP_COL = "id_configuracao"
RANDOM_STATE = 42

# Caminhos relativos (a partir de scripts/)
DATA_PATH = os.path.join("..", "dados", "banco_CCO_XAI_final_500_arvores.csv")
MODELS_DIR = os.path.join("..", "modelos")
TABLES_DIR = os.path.join("..", "resultados", "tabelas")
FIGURES_DIR = os.path.join("..", "resultados", "figuras")


# ── Funções auxiliares ───────────────────────────────────────────────────────

def garantir_diretorios():
    """Cria os diretórios de saída caso não existam."""
    for d in [MODELS_DIR, TABLES_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)


def carregar_dados():
    """Carrega o CSV e retorna X, dicionário de y (por alvo) e groups."""
    print("=" * 70)
    print("1. CARREGAMENTO DOS DADOS")
    print("=" * 70)
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    print(f"   Dimensões do DataFrame: {df.shape}")
    print(f"   Features: {FEATURES}")
    print(f"   Alvos:    {TARGETS}")
    print(f"   Coluna de agrupamento: {GROUP_COL}")
    print(f"   Configurações únicas:  {df[GROUP_COL].nunique()}")
    return df


def dividir_treino_teste(df):
    """
    Divisão treino/teste (~80/20) por configuração usando GroupShuffleSplit.
    Todas as 5 sementes de uma configuração ficam no mesmo conjunto.
    Retorna X_train, X_test, dicts de y_train/y_test, groups_train/test.
    """
    print("\n" + "=" * 70)
    print("2. DIVISÃO TREINO / TESTE (GroupShuffleSplit)")
    print("=" * 70)

    X = df[FEATURES]
    groups = df[GROUP_COL]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    # Dicionário com os alvos separados
    y_train = {}
    y_test = {}
    for target in TARGETS:
        y_train[target] = df[target].iloc[train_idx]
        y_test[target] = df[target].iloc[test_idx]

    # Estatísticas da divisão
    configs_train = groups_train.nunique()
    configs_test = groups_test.nunique()
    print(f"   Treino : {configs_train} configurações  |  {len(train_idx)} amostras")
    print(f"   Teste  : {configs_test} configurações  |  {len(test_idx)} amostras")

    # Salvar tabela de divisão
    divisao = pd.DataFrame({
        GROUP_COL: df[GROUP_COL].unique()
    })
    train_configs = set(groups_train.unique())
    divisao["conjunto"] = divisao[GROUP_COL].apply(
        lambda c: "train" if c in train_configs else "test"
    )
    divisao = divisao.sort_values(GROUP_COL).reset_index(drop=True)
    path_divisao = os.path.join(TABLES_DIR, "divisao_train_test.csv")
    divisao.to_csv(path_divisao, index=False, encoding="utf-8")
    print(f"   Tabela de divisão salva em: {path_divisao}")

    return X_train, X_test, y_train, y_test, groups_train, groups_test


def salvar_dados_split(X_train, X_test, y_train, y_test, groups_train, groups_test):
    """Serializa os dados da divisão treino/teste com joblib."""
    print("\n   Salvando dados de treino/teste com joblib...")
    joblib.dump(X_train, os.path.join(MODELS_DIR, "X_train.joblib"))
    joblib.dump(X_test, os.path.join(MODELS_DIR, "X_test.joblib"))
    joblib.dump(y_train["volume_mm3"], os.path.join(MODELS_DIR, "y_train_volume.joblib"))
    joblib.dump(y_test["volume_mm3"], os.path.join(MODELS_DIR, "y_test_volume.joblib"))
    joblib.dump(y_train["raio_root_mm"], os.path.join(MODELS_DIR, "y_train_raio.joblib"))
    joblib.dump(y_test["raio_root_mm"], os.path.join(MODELS_DIR, "y_test_raio.joblib"))
    joblib.dump(groups_train, os.path.join(MODELS_DIR, "groups_train.joblib"))
    joblib.dump(groups_test, os.path.join(MODELS_DIR, "groups_test.joblib"))
    print("   ✔ Dados de treino/teste salvos em ../modelos/")


def treinar_dummy(X_train, y_train):
    """Treina um DummyRegressor (baseline) com estratégia de média."""
    modelo = DummyRegressor(strategy="mean")
    modelo.fit(X_train, y_train)
    return modelo, {"strategy": "mean"}, None


def treinar_linear_regression(X_train, y_train):
    """Treina uma Regressão Linear simples (sem ajuste de hiperparâmetros)."""
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo, {"modelo": "LinearRegression"}, None


def treinar_decision_tree(X_train, y_train, groups_train):
    """
    Treina uma Árvore de Decisão com busca em grade (GridSearchCV)
    para max_depth, usando GroupKFold(n_splits=5).
    """
    param_grid = {"max_depth": [3, 5, 7, 10, 15, None]}
    gkf = GroupKFold(n_splits=5)
    gs = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=gkf,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        refit=True,
    )
    gs.fit(X_train, y_train, groups=groups_train)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def treinar_random_forest(X_train, y_train, groups_train):
    """
    Treina um Random Forest com RandomizedSearchCV (n_iter=20),
    usando GroupKFold(n_splits=5).
    """
    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_leaf": [2, 5, 10],
    }
    gkf = GroupKFold(n_splits=5)
    rs = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=RANDOM_STATE),
        param_distributions=param_distributions,
        n_iter=20,
        cv=gkf,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )
    rs.fit(X_train, y_train, groups=groups_train)
    return rs.best_estimator_, rs.best_params_, rs.best_score_


def treinar_xgboost(X_train, y_train, groups_train):
    """
    Treina um XGBRegressor com RandomizedSearchCV (n_iter=20),
    usando GroupKFold(n_splits=5).
    """
    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    gkf = GroupKFold(n_splits=5)
    rs = RandomizedSearchCV(
        estimator=XGBRegressor(
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=1,
        ),
        param_distributions=param_distributions,
        n_iter=20,
        cv=gkf,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )
    rs.fit(X_train, y_train, groups=groups_train)
    return rs.best_estimator_, rs.best_params_, rs.best_score_


def treinar_todos_modelos(X_train, y_train, groups_train, target_name):
    """
    Treina os 5 modelos para um dado alvo e retorna um dicionário
    com {nome_modelo: (modelo_treinado, best_params, cv_score)}.
    """
    print(f"\n   --- Alvo: {target_name} ---")
    resultados = {}

    # (a) Dummy
    print(f"   [1/5] DummyRegressor (baseline)...")
    modelo, params, score = treinar_dummy(X_train, y_train)
    resultados["dummy"] = (modelo, params, score)
    print(f"         Parâmetros: {params}")

    # (b) Regressão Linear
    print(f"   [2/5] LinearRegression...")
    modelo, params, score = treinar_linear_regression(X_train, y_train)
    resultados["linear_regression"] = (modelo, params, score)
    print(f"         Parâmetros: {params}")

    # (c) Árvore de Decisão
    print(f"   [3/5] DecisionTreeRegressor (GridSearchCV)...")
    modelo, params, score = treinar_decision_tree(X_train, y_train, groups_train)
    resultados["decision_tree"] = (modelo, params, score)
    print(f"         Melhores parâmetros: {params}")
    print(f"         Melhor MAE (CV): {-score:.4f}")

    # (d) Random Forest
    print(f"   [4/5] RandomForestRegressor (RandomizedSearchCV)...")
    modelo, params, score = treinar_random_forest(X_train, y_train, groups_train)
    resultados["random_forest"] = (modelo, params, score)
    print(f"         Melhores parâmetros: {params}")
    print(f"         Melhor MAE (CV): {-score:.4f}")

    # (e) XGBoost
    print(f"   [5/5] XGBRegressor (RandomizedSearchCV)...")
    modelo, params, score = treinar_xgboost(X_train, y_train, groups_train)
    resultados["xgboost"] = (modelo, params, score)
    print(f"         Melhores parâmetros: {params}")
    print(f"         Melhor MAE (CV): {-score:.4f}")

    return resultados


def salvar_modelos(resultados_por_alvo):
    """
    Serializa todos os modelos treinados em ../modelos/
    com nomenclatura: {nome_modelo}_{nome_alvo}.joblib
    """
    print("\n" + "=" * 70)
    print("4. SERIALIZAÇÃO DOS MODELOS")
    print("=" * 70)
    for target_name, resultados in resultados_por_alvo.items():
        for model_name, (modelo, _, _) in resultados.items():
            filename = f"{model_name}_{target_name}.joblib"
            filepath = os.path.join(MODELS_DIR, filename)
            joblib.dump(modelo, filepath)
            print(f"   ✔ {filepath}")


def salvar_hiperparametros(resultados_por_alvo):
    """
    Consolida e salva os melhores hiperparâmetros de cada modelo/alvo
    em ../resultados/tabelas/hiperparametros.csv.
    """
    print("\n" + "=" * 70)
    print("5. HIPERPARÂMETROS — TABELA CONSOLIDADA")
    print("=" * 70)
    registros = []
    for target_name, resultados in resultados_por_alvo.items():
        for model_name, (_, params, cv_score) in resultados.items():
            registro = {
                "alvo": target_name,
                "modelo": model_name,
                "cv_neg_mae": cv_score if cv_score is not None else np.nan,
            }
            # Adicionar cada hiperparâmetro como coluna
            for k, v in params.items():
                registro[k] = v
            registros.append(registro)

    df_params = pd.DataFrame(registros)
    path_params = os.path.join(TABLES_DIR, "hiperparametros.csv")
    df_params.to_csv(path_params, index=False, encoding="utf-8")
    print(f"   Tabela salva em: {path_params}")
    print(df_params.to_string(index=False))


def imprimir_resumo(resultados_por_alvo):
    """Imprime um resumo final com os melhores modelos e scores."""
    print("\n" + "=" * 70)
    print("7. RESUMO FINAL")
    print("=" * 70)
    for target_name, resultados in resultados_por_alvo.items():
        print(f"\n   Alvo: {target_name}")
        print(f"   {'Modelo':<22} {'MAE (CV)':>12}  Melhores Parâmetros")
        print(f"   {'-'*22} {'-'*12}  {'-'*40}")
        for model_name, (_, params, cv_score) in resultados.items():
            if cv_score is not None:
                mae_str = f"{-cv_score:>12.4f}"
            else:
                mae_str = f"{'N/A':>12}"
            print(f"   {model_name:<22} {mae_str}  {params}")


# ── Execução principal ───────────────────────────────────────────────────────

def main():
    """Pipeline principal de treinamento de modelos."""
    # Ajustar diretório de trabalho para scripts/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    garantir_diretorios()

    # 1. Carregar dados
    df = carregar_dados()

    # 2. Divisão treino/teste
    X_train, X_test, y_train, y_test, groups_train, groups_test = dividir_treino_teste(df)

    # Salvar dados da divisão
    salvar_dados_split(X_train, X_test, y_train, y_test, groups_train, groups_test)

    # 3. Treinar modelos para cada alvo
    print("\n" + "=" * 70)
    print("3. TREINAMENTO DOS MODELOS")
    print("=" * 70)

    resultados_por_alvo = {}
    for target in TARGETS:
        resultados_por_alvo[target] = treinar_todos_modelos(
            X_train, y_train[target], groups_train, target
        )

    # 4. Salvar modelos
    salvar_modelos(resultados_por_alvo)

    # 5. Salvar hiperparâmetros
    salvar_hiperparametros(resultados_por_alvo)

    # 7. Resumo final
    imprimir_resumo(resultados_por_alvo)

    print("\n" + "=" * 70)
    print("✔ Pipeline de modelos concluído com sucesso!")
    print("=" * 70)


if __name__ == "__main__":
    main()
