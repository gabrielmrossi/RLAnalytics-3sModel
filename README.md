# RLAnalytics-3sModel

Análise de telemetria e modelagem comportamental para partidas 3v3 de Rocket League (TCC).

## 📌 Sobre o Projeto

Este projeto analisa replays de partidas de Rocket League no modo 3v3 (Standard) para avaliar posicionamento tático, rotações e métricas de desempenho de jogadores e equipes através de técnicas de telemetria e Machine Learning.

### Principais Componentes
- **Parser (`src/parser.py`)**: Integração com `rrrocket.exe` para converter arquivos binários de replay (`.replay`) em estruturas JSON.
- **Analyzer (`src/analyzer.py`)**: Extração de métricas espaciais e temporais (velocidade média, boost, dispersão da equipe, área do triângulo tático, cobertura defensiva, etc.).
- **Feature Engineering (`src/feature_engineering.py`)**: Agregação de features por partida e por equipe para modelagem supervisionada e não supervisionada.
- **Modelos (`src/knn_model.py`)**:
  - Classificação (KNN com LOO-CV, Decision Tree, Random Forest).
  - Clusterização comportamental não supervisionada (K-Means com PCA).
  - Detecção de erros de posicionamento e benchmarking comparativo.
- **Dashboard Streamlit (`src/streamlit_app.py`)**: Interface interativa para visualização de métricas, clusters e análises das partidas.

---

## 🚀 Como Executar

### Pré-requisitos
- Python 3.10+
- Ferramenta `bin/rrrocket.exe` (inclusa no repositório)

### Instalação

```bash
# Clone o repositório
git clone https://github.com/<SEU_USUARIO>/RLAnalytics-3sModel.git
cd RLAnalytics-3sModel

# Crie e ative o ambiente virtual
python -m venv .venv
# No Windows PowerShell:
.venv\Scripts\Activate.ps1

# Instale as dependências
pip install -r requirements.txt
```

### Executando a Aplicação Streamlit

```bash
streamlit run src/streamlit_app.py
```
