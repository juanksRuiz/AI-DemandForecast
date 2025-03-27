import joblib
from pathlib import Path
import json

import numpy as np
import pandas as pd
from .utility_funcs import *

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

# Carga del modelo desde el archivo
model_path = f"{BASE_DIR}/forecast_model_with_exogenous-{__version__}.joblib"
loaded_model = joblib.load(model_path)

def predict(test_steps):
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
        ultima_fecha = pd.to_datetime(metadata['ultima_fecha'])
    
    X_exog_test = generate_exogenous(ultima_fecha + pd.Timedelta(days=1), test_steps)
    predictions = loaded_model.forecast(steps=test_steps)
    return predictions
