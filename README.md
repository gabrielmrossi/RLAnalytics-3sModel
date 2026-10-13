# RLAnalytics-3sModel

Análise de telemetria e modelagem comportamental para partidas 3v3 de Rocket League (TCC).

---

## 📌 Sobre o Projeto

Este projeto analisa replays de partidas de Rocket League no modo 3v3 (Standard) para avaliar posicionamento tático, rotações e métricas de desempenho de jogadores e equipes através de técnicas de telemetria e Machine Learning.

O pipeline abrange desde o parsing binário do replay até a extração de métricas cinemáticas/espaciais, treinamento de modelos supervisionados (KNN, Decision Tree, Random Forest) e não supervisionados (K-Means com PCA), detecção de anomalias posicionais e visualização em dashboard interativo ou figuras acadêmicas de alta resolução (300 DPI).

---

## 🏗️ Arquitetura do Repositório

```text
RLAnalytics-3sModel/
├── app/                        # Dashboard Streamlit Modular
│   ├── main.py                 # Ponto de entrada da interface web
│   ├── theme.py                # Estilização CSS e configuração de layout
│   ├── services.py             # Serviços de dados em cache e orquestração
│   ├── components/             # Componentes reutilizáveis (campo 2D, downloads)
│   └── tabs/                   # Abas independentes (Live, Rank, Modelos, Erros)
├── artifacts/
│   └── figures/                # Figuras exportadas em 300 DPI para artigos/TCC
├── bin/
│   └── rrrocket.exe            # Parser binário de replays de Rocket League
├── data/
│   ├── raw/                    # Arquivos .replay brutos
│   └── processed/              # JSONs de telemetria, CSVs e modelos joblib
├── scripts/                    # Scripts utilitários e CLIs
│   ├── build_dataset.py        # Extração de features de replays para dataset
│   ├── train_models.py         # Pipeline de treinamento e validação de modelos
│   └── export_figures.py       # Gerador de gráficos acadêmicos de alta fidelidade
├── src/                        # Pacote Python principal (rl_analytics)
│   ├── analyzer.py             # Extração de telemetria vetorial e métricas espaciais
│   ├── benchmarking.py         # Comparativo com Cross-Validation e LOO-CV
│   ├── classifiers.py          # Classificadores de rank (KNN, DT, RF)
│   ├── clustering.py           # Agrupamento tático (K-Means + PCA)
│   ├── config.py               # Configurações centralizadas, paleta e caminhos
│   ├── error_detection.py      # Detectores heurístico e orientado a modelo
│   ├── feature_catalog.py      # Catálogo central de features, labels e narrativas
│   ├── feature_engineering.py  # Agregação temporal e formação tática
│   ├── knn_model.py            # Fachada unificada retrocompatível
│   ├── parser.py               # Integração subprocess com rrrocket
│   ├── streamlit_app.py        # Fachada retrocompatível do Streamlit
│   └── visualization/          # Renderizadores de gráficos
│       ├── interactive.py      # Gráficos interativos (Plotly)
│       └── static.py           # Gráficos estáticos (Matplotlib Agg)
├── tests/
│   ├── fixtures/               # Baselines numéricas e hashes de integridade
│   └── test_baselines.py       # Testes de caracterização e paridade lógica
├── pyproject.toml              # Especificação de build e console scripts
├── requirements.lock           # Versões exatas das dependências congeladas
└── requirements.txt            # Dependência com instalação editável (-e .)
```

---

## 🚀 Como Executar

### Pré-requisitos
- Python 3.10+ (testado em Python 3.13)
- Windows (utiliza `bin/rrrocket.exe` incluído)

### Instalação

Recomenda-se utilizar um ambiente virtual:

```bash
# Clone o repositório e navegue até a pasta
git clone https://github.com/gabrielmrossi/RLAnalytics-3sModel.git
cd RLAnalytics-3sModel

# Crie o ambiente virtual
python -m venv .venv

# No Windows PowerShell:
.venv\Scripts\Activate.ps1

# Instale o pacote em modo editável com as dependências
pip install -e .
```

---

## 🛠️ Comandos CLI

O pacote expõe ferramentas de linha de comando diretamente pelo ambiente virtual:

### 1. Construir Dataset de Telemetria
Processa os replays brutos em `data/raw/` e gera o dataset processado:
```bash
rl-build-dataset
# ou: python scripts/build_dataset.py
```

### 2. Treinar Modelos de Machine Learning
Executa validação cruzada, avalia modelos de classificação e salva os modelos `.joblib` em `data/processed/`:
```bash
rl-train
# ou: python scripts/train_models.py
```

### 3. Exportar Figuras Acadêmicas (300 DPI)
Gera gráficos estáticos de alta resolução em `artifacts/figures/` sem depender da interface gráfica:
```bash
rl-export-figures
# ou: python scripts/export_figures.py
```

---

## 📊 Dashboard Interativo (Streamlit)

Para iniciar o dashboard tático interativo:

```bash
streamlit run app/main.py
```

> **Nota de compatibilidade:** O comando legado `streamlit run src/streamlit_app.py` continua funcionando como fachada para o novo dashboard modular.

### Recursos do Dashboard
- **Replay Live**: Visualização 2D do campo em tempo real, calor de posições, posse de boost e eventos (gols, demolições).
- **Higher vs. Lower Rank**: Comparativo das distribuições táticas de equipes de rank alto versus rank baixo.
- **Modelos de ML**: Avaliação de KNN, Random Forest, Decision Tree e clusters comportamentais com PCA.
- **Erros Táticos**: Diagnóstico de anomalias posicionais da equipe com score tático e narrativas interpretáveis.
- **Exportação Direta**: Download em memória de relatórios e figuras sem causar concorrência de I/O em disco.

---

## 🧪 Testes Automatizados e Garantia de Integridade

O repositório possui uma suíte de caracterização que valida 9 baselines numéricas e estruturais (paridade de acurácia, matrizes de confusão, importâncias de features, clusters e hashes de telemetria):

```bash
python -m unittest tests/test_baselines.py
```
