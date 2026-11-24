# 🌳 Projeto Integrador CCO-XAI — Plataforma CCO/UFOP

**Modelagem Preditiva e Explicabilidade em Árvores Arteriais Sintéticas**  
*Departamento de Computação (DECOM) — Universidade Federal de Ouro Preto (UFOP)*  
*Disciplina: Inteligência Artificial Explicável (XAI) — 2026*  
*Professor: Prof. Rafael Alves Bonfim de Queiroz*

---

## 📖 Visão Geral

Este projeto investiga como os parâmetros de entrada hemodinâmicos e estruturais do método **Constrained Constructive Optimization (CCO)** se relacionam com o **volume intravascular total ($V_{\rm intra}$)** e o **raio do segmento proximal raiz ($r_{\rm root}$)** de 500 árvores arteriais sintéticas (100 configurações com 5 sementes cada).

### 🎯 Variáveis do Estudo

| Tipo | Variável no CSV | Descrição Física | Unidade |
| :--- | :--- | :--- | :--- |
| **Entrada** | `deltaP_mmHg` | Queda de pressão ($\Delta P$) | mmHg |
| **Entrada** | `gamma` | Expoente de bifurcação de Murray ($\gamma$) | adimensional |
| **Entrada** | `Q_ml_min` | Vazão total de perfusão ($Q_{\rm perf}$) | mL/min |
| **Entrada** | `Nterm` | Número de segmentos terminais ($N_{\rm term}$) | inteiro |
| **Alvo 1** | `volume_mm3` | Volume intravascular total ($V_{\rm intra}$) | mm³ |
| **Alvo 2** | `raio_root_mm` | Raio do segmento proximal raiz ($r_{\rm root}$) | mm |

> **Controle de Vazamento:** Apenas as 4 entradas acima são utilizadas nos modelos preditivos. O identificador `id_configuracao` é utilizado estritamente para agrupar as divisões de validação e teste (`GroupShuffleSplit` / `GroupKFold`), garantindo que todas as 5 sementes de uma mesma árvore permaneçam no mesmo subconjunto.

---

## 🗂️ Estrutura de Diretórios

```
TP/
├── dados/
│   └── banco_CCO_XAI_final_500_arvores.csv   # Dataset oficial (500 árvores)
├── scripts/
│   ├── 01_eda.py                             # Análise exploratória de dados
│   ├── 02_modelos.py                         # Treinamento e ajuste dos modelos
│   └── 03_avaliacao.py                       # Métricas, resíduos e gráficos
├── modelos/                                  # Modelos serializados (.joblib)
├── resultados/
│   ├── figuras/                              # Gráficos em alta resolução (DPI=150)
│   └── tabelas/                              # Métricas e resumos (.csv e .tex)
├── webapp/                                   # Dashboard Web interativo (Streamlit)
│   ├── app.py                                # Ponto de entrada do dashboard
│   └── pages/
│       ├── 1_📊_EDA_Interactivo.py           # Filtros dinâmicos e histogramas
│       ├── 2_🤖_Predictor.py                 # Sliders e inferência em tempo real
│       ├── 3_🔍_XAI_Explorer.py              # Explicabilidade SHAP/LIME (Parte II)
│       └── 4_📈_Comparador.py                # Resíduos, Obs vs Pred e métricas
├── template-relatorio/                       # Template LaTeX do relatório preliminar
├── template-apresentacao/                    # Template LaTeX da apresentação Beamer
├── requirements.txt                          # Dependências com versões exatas
└── README.md                                 # Este guia de execução
```

---

## ⚙️ Pré-requisitos e Instalação

* **Python:** 3.10 ou superior (testado com Python 3.13)
* **Compilador LaTeX:** MiKTeX ou TeX Live (para compilação dos relatórios `.tex`)

Para instalar as dependências necessárias:

```bash
pip install -r requirements.txt
```

---

## 🚀 Como Executar o Pipeline

Todos os scripts utilizam caminhos relativos e semente fixa (`random_state=42`) para garantir **reprodutibilidade estrita**.

### 1. Análise Exploratória (EDA)
Gera estatísticas descritivas, matriz de correlação de Pearson, histogramas e estudo de variabilidade estocástica:
```bash
python scripts/01_eda.py
```
*Saídas em:* `resultados/figuras/` e `resultados/tabelas/`

### 2. Treinamento e Validação dos Modelos
Executa a divisão agrupada (80% treino / 20% teste por `id_configuracao`), busca de hiperparâmetros com `GroupKFold(5)` e treina os 5 algoritmos (*Dummy, Regressão Linear, Árvore de Decisão, Random Forest e XGBoost*) para cada target:
```bash
python scripts/02_modelos.py
```
*Saídas em:* `modelos/*.joblib` e `resultados/tabelas/hiperparametros.csv`

### 3. Avaliação e Comparação de Desempenho
Calcula MAE, RMSE e $R^2$ em treino e teste, gera dispersões de Observado vs Previsto com linha de identidade e gráficos de resíduos:
```bash
python scripts/03_avaliacao.py
```
*Saídas em:* `resultados/tabelas/metricas_modelos.csv` e figuras comparativas.

---

## 🌐 Dashboard Web Interativo (Streamlit)

Para iniciar o painel analítico e interagir com sliders de previsão e visualizações dinâmicas:

```bash
streamlit run webapp/app.py
```

O aplicativo abrirá automaticamente no navegador em `http://localhost:8501`.

---

## 📄 Compilação do Relatório e da Apresentação em LaTeX

A partir da raiz do repositório, compile o relatório técnico com `latexmk` (o diretório já inclui `latexmkrc`):
```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error template-relatorio/template-relatorio/template_artigo_portugues.tex
```

Para compilar a apresentação Beamer:
```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error template-apresentacao/template-apresentacao/template_apresentacao_beamer.tex
```

Os templates leem as imagens de suas próprias pastas `figuras/`. Após regenerar os gráficos em `resultados/figuras/`, copie os PNGs atualizados para `template-relatorio/template-relatorio/figuras/` e `template-apresentacao/template-apresentacao/figuras/` antes de recompilar.

---

## 👥 Autoria e Afiliação

* **Instituição:** Universidade Federal de Ouro Preto (UFOP)
* **Departamento:** Departamento de Computação (DECOM)
* **Plataforma:** CCO/UFOP — Modelagem Computacional Vascular

