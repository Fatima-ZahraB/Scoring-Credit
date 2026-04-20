# creditscoring/utils.py
from pathlib import Path
import pandas as pd

def load_csv(filename: str) -> pd.DataFrame:
    # Chemin vers le dossier data dans le package
    data_path = Path(__file__).parent / "data" / "raw_data" / filename
    return pd.read_csv(data_path)
#=============================================================================


def save_csv(df: pd.DataFrame, filename: str) -> None:
    data_path = Path(__file__).parent / "data" / "processed_data"
    # créer le dossier s'il n'existe pas
    data_path.mkdir(parents=True, exist_ok=True)
    file_path = data_path / filename
    df.to_csv(file_path, index=False)


#=============================================================================

def save_outputs(df: pd.DataFrame, filename: str) -> None:
    data_path = Path(__file__).parent / "data" / "predictions"
    # créer le dossier s'il n'existe pas
    data_path.mkdir(parents=True, exist_ok=True)
    file_path = data_path / filename
    df.to_csv(file_path, index=False)