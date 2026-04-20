import joblib
import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "best_pipe_lgbm.pkl"
DATA_PATH  = Path(__file__).parent.parent / "data" / "predictions" /  "sample_deploy.csv"


class Predictor:
    def __init__(self):
        self.model = None
        self.data: pd.DataFrame = None
        self.model_loaded = False
        self.data_loaded = False

    def load(self):
        """Load model and client data at startup."""
        # Load model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        self.model = joblib.load(MODEL_PATH)
        self.model_loaded = True
        logger.info(f"Model loaded from {MODEL_PATH}")

        # Load client data
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Data not found at {DATA_PATH}")
        self.data = pd.read_csv(DATA_PATH)

        # Drop TARGET if present
        if "TARGET" in self.data.columns:
            self.data = self.data.drop(columns=["TARGET"])

        self.data_loaded = True
        logger.info(f"Data loaded: {len(self.data)} clients from {DATA_PATH}")

    def predict(self, sk_id: int) -> Tuple[float, float]:
        """
        Returns (probability_of_default, inference_time_ms).
        Raises ValueError if SK_ID_CURR not found.
        """
        if not self.model_loaded or not self.data_loaded:
            raise RuntimeError("Predictor not initialized. Call load() first.")

        # Lookup client
        client_row = self.data[self.data["SK_ID_CURR"] == sk_id]
        if client_row.empty:
            raise ValueError(f"SK_ID_CURR {sk_id} not found in dataset")

        # Drop ID column before inference
        X_client = client_row.drop(columns=["SK_ID_CURR"], errors="ignore")

        # Inference + timing
        start = time.perf_counter()
        proba = self.model.predict_proba(X_client)[0, 1]
        elapsed_ms = (time.perf_counter() - start) * 1000

        return float(proba), elapsed_ms

    @property
    def total_clients(self) -> int:
        if self.data is None:
            return 0
        return len(self.data)


# Singleton instance
predictor = Predictor()
