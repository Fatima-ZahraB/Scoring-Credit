# 🏦 Home Credit Default Risk — Scoring API

Prédiction de probabilité de défaut de remboursement d'un crédit bancaire.
Cette API permet d’évaluer la **solvabilité d’un client** pour statuer sur une demande de prêt bancaire.  
Elle s’appuie sur un modèle de **Machine Learning** pour prédire si un client est **solvable** ou **défaillant**, à partir de ses données personnelles,  socio-économiques et de son profil.


[![CI/CD](https://github.com/Fatima-ZahraB/Scoring-Credit/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/Fatima-ZahraB/Scoring-Credit/actions)
![Python](https://img.shields.io/badge/python-3.12%20|%20CPython-blue?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker&logoColor=white)

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
├── logs/
│   └── log_predictions.csv  # logs
├── models/
│   └── best_pipe_lgbm.pkl  # Modèle LightGBM entraîné
├── data/
│   └── predictions       # Historique des inputs et outputs de prédictions
│   └── processed_data    # Dataset de train et test prêt pour la modélisation
│   └── raw_data          # Ensemble des datasets du projet ( 8 fichiers csv )
├── Dockerfile
├── start.sh
├── requirements.txt
└── .github/workflows/ci_cd.yml
```

---

## 🚀 Fonctionnalités

- ✅ **Endpoint principal `/predict`** pour obtenir une prédiction de solvabilité.  
- 🧩 **Validation stricte des données** via des modèles `Pydantic`.  
- 🧾 **Logs structurés** dans `logs/`.  
- 🧪 **Tests unitaires et d’intégration** (via `pytest`).  
- 🐳 **Image Docker** prête à être déployée.

---

## Démarrage rapide

### 1. Installation

```bash
git clone https://github.com/Fatima-ZahraB/Scoring-Credit.git
cd creditscoring
pip install -r requirements.txt
```
ou 

### 2. Étapes d'installation avec POETRY

1. **Cloner le repository**
```bash
git clone https://github.com/Fatima-ZahraB/Scoring-Credit.git
cd creditscoring
```

2. **Installer les dépendances**
```bash
poetry install
```

3. **Activer l'environnement virtuel**
```bash
poetry shell
```

4. **Vérifier l'installation**
```bash
python --version  # Devrait afficher Python 3.12.x
```
### 2. Prérequis

Placer les fichiers suivants avant de démarrer :

```
models/best_pipe_lgbm.pkl               modèle entraîné (joblib)
data/predictions/sample_deploy.csv      CSV des clients (avec SK_ID_CURR)
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


### 🧩 Endpoints disponibles

| Méthode | Route        | Description |
|----------|--------------|-------------|
| `GET`    | `/`          | Page d’accueil |
| `POST`   | `/predict`   | Prédiction de défaut |
| `GET`    | `/logs`      | Lecture des logs |


---

## Tests

Les tests unitaires et d’intégration sont gérés avec **pytest** et **pytest-cov** afin de garantir la fiabilité du modèle et de mesurer la couverture du code.  

### ▶️ Lancer la suite de tests (depuis la racine du projet)  

```bash
# Lancer tous les tests
pytest tests/ -v

# Avec rapport de couverture
pytest tests/ --cov=app --cov-report=term-missing
```

---

## 🧰 Dockerisation

La dockerisation permet d’encapsuler l’API FastAPI et son modèle de Machine Learning dans un conteneur léger et portable.
Cela garantit une exécution identique sur tous les environnements (local, cloud, CI/CD) et simplifie le déploiement.

### 🎯 Objectif

Faciliter le déploiement de l’API sans dépendances locales.

Garantir un environnement reproductible entre les machines de développement, test et production.

Simplifier le scaling horizontal (plusieurs instances conteneurisées derrière un load balancer).

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
