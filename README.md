# 🏦 Home Credit Default Risk — Scoring API

Prédiction de probabilité de défaut de remboursement d'un crédit bancaire.

[![CI/CD](https://github.com/YOUR_USERNAME/creditscoring/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/creditscoring/actions)

---

## Architecture

```
creditscoring/
├── app/
│   ├── main.py          # FastAPI — endpoints /predict /health /logs
│   ├── schemas.py       # Validation Pydantic
│   ├── predictor.py     # Chargement modèle + inférence
│   └── logger.py        # Logging CSV des prédictions
├── streamlit_app.py     # Interface UI + Dashboard monitoring
├── monitoring/
│   └── drift_report.py  # Rapport Evidently (data drift)
├── tests/
│   └── test_api.py      # Tests automatisés (pytest)
├── data/
│   └── clients_deploy.csv  # Dataset de déploiement
├── models/
│   └── best_pipe_lgbm.pkl  # Modèle LightGBM entraîné
├── logs/
│   └── predictions       # Historique des inputs et outputs de prédictions
│   └── processed_data    # Dataset de train et test prêt pour la modélisation
│   └── raw_data          # Ensemble des datasets du projet ( 8 fichiers csv )
├── Dockerfile
├── start.sh
├── requirements.txt
└── .github/workflows/ci_cd.yml
```

---

## Démarrage rapide

### 1. Installation

```bash
git clone https://github.com/Fatima-ZahraB/creditscoring.git
cd creditscoring
pip install -r requirements.txt
```

### 2. Prérequis

Placer les fichiers suivants avant de démarrer :

```
models/best_pipe_lgbm.pkl   ← modèle entraîné (joblib)
data/clients_deploy.csv     ← CSV des clients (avec SK_ID_CURR)
```

### 3. Lancer l'API FastAPI

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Documentation interactive : [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Lancer Streamlit

```bash
streamlit run streamlit_app.py
```

Interface : [http://localhost:8501](http://localhost:8501)

### 5. Avec Docker

```bash
# Build
docker build -t credit-scoring .

# Run
docker run -p 8000:8000 -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  credit-scoring
```

---

## API

### `POST /predict`

Retourne la probabilité de défaut pour un client.

**Request**
```json
{
  "SK_ID_CURR": 100001
}
```

**Response**
```json
{
  "SK_ID_CURR": 100001,
  "probability_default": 0.142857,
  "risk_level": "LOW",
  "recommendation": "Credit likely to be approved",
  "inference_time_ms": 3.21,
  "timestamp": "2024-01-15T14:32:10.123456"
}
```

| Risk Level | Probabilité | Recommandation |
|---|---|---|
| LOW | < 30% | Credit likely to be approved |
| MEDIUM | 30–60% | Manual review recommended |
| HIGH | > 60% | Credit likely to be refused |

### `GET /health`

Vérification de l'état de l'API.

### `GET /logs`

Retourne l'historique de toutes les prédictions.

---

## Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Avec rapport de couverture
pytest tests/ --cov=app --cov-report=term-missing
```

---

## Monitoring

### Dashboard Streamlit

L'onglet **Monitoring** de l'app Streamlit affiche :
- Distribution des scores prédits
- Taux de risque (LOW / MEDIUM / HIGH)
- Latence d'inférence (moyenne + P95)
- Historique des 50 dernières prédictions
- Rapport Evidently de data drift (si généré)

### Générer un rapport Evidently

```bash
python monitoring/drift_report.py
```

Le rapport HTML est sauvegardé dans `logs/drift_report.html`.

---

## CI/CD — GitHub Actions

Le pipeline se déclenche à chaque push sur `main` :

1. **Tests** — pytest avec rapport de couverture
2. **Build Docker** — validation de l'image
3. **Deploy** — push automatique vers Hugging Face Spaces

### Configuration des secrets GitHub

Dans `Settings > Secrets and variables > Actions` :

| Secret | Description |
|---|---|
| `HF_TOKEN` | Token Hugging Face (Settings > Access Tokens) |
| `HF_USERNAME` | Ton username Hugging Face |

### Configurer le nom du Space

Dans `.github/workflows/ci_cd.yml`, modifier :
```yaml
env:
  HF_SPACE: ${{ secrets.HF_USERNAME }}/credit-scoring
```

---

## Modèle

- **Algorithme** : LightGBM (pipeline sklearn)
- **Métrique** : ROC AUC = **0.7864** (OOF, 5 folds)
- **Features** : 124 numériques + 15 catégorielles (après feature engineering)
- **Tuning** : Optuna (TPE Sampler, 20 trials)

---

## Stack technique

| Composant | Technologie |
|---|---|
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| ML | LightGBM + scikit-learn |
| UI | Streamlit |
| Monitoring | Evidently |
| Tests | pytest + pytest-cov |
| Conteneurisation | Docker |
| CI/CD | GitHub Actions |
| Déploiement | Hugging Face Spaces |
