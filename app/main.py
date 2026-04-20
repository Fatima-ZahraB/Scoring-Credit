import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import predictor
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.logger import log_prediction

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model et data...")
    predictor.load()
    logger.info(f"Pret. {predictor.total_clients} clients loaded.")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="API de prédiciton du risque de défaut bancaire",
    description="Prédire la probabilité de défaut ou de remboursement du crédit pour chaque demande client à partir de son identifiant .",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Check API and model status."""
    return HealthResponse(
        status="ok",
        model_loaded=predictor.model_loaded,
        data_loaded=predictor.data_loaded,
        total_clients=predictor.total_clients,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Prediction"],
)
def predict(request: PredictionRequest):
    """
    Predict the probability of default for a client.

    - **SK_ID_CURR**: unique client identifier (must exist in the deployed dataset)
    """
    try:
        proba, inference_ms = predictor.predict(request.SK_ID_CURR)
        response = PredictionResponse.from_proba(
            sk_id=request.SK_ID_CURR,
            proba=proba,
            inference_time_ms=inference_ms,
        )
        log_prediction(
            sk_id=request.SK_ID_CURR,
            probability=proba,
            risk_level=response.risk_level,
            inference_time_ms=inference_ms,
            status="success",
        )
        logger.info(
            f"Prediction OK | SK_ID={request.SK_ID_CURR} | "
            f"proba={proba:.4f} | risk={response.risk_level} | "
            f"{inference_ms:.1f}ms"
        )
        return response

    except ValueError as e:
        log_prediction(
            sk_id=request.SK_ID_CURR,
            status="error",
            error_message=str(e),
        )
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        log_prediction(
            sk_id=request.SK_ID_CURR,
            status="error",
            error_message=str(e),
        )
        logger.error(f"Unexpected error for SK_ID={request.SK_ID_CURR}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/logs", tags=["Monitoring"])
def get_logs():
    """Return all prediction logs."""
    from app.logger import read_logs
    return read_logs()
