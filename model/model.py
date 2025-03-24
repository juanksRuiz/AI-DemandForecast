import joblib
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

# Carga del modelo desde el archivo
loaded_model = joblib.load(f"{BASE_DIR}/forecast_model_with_exogenous-{__version__}.joblib")

def predict(txt_date):
    date = pd.Timedelta(txt_date)
    
