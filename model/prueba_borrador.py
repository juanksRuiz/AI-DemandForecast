import json
import os
import pandas as pd

from utility_funcs import *
# Prueba de la funcion de generacion de variables exogenas steps y fecha inicial

metadata_path = os.path.join(os.getcwd(), 'metadata.json')
with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        ultima_fecha = pd.to_datetime(metadata['ultima_fecha'])
    
test_steps = 3
X_exog_test = generate_exogenous(ultima_fecha + pd.Timedelta(days=1), test_steps)

print(type(X_exog_test))
print(pd.DataFrame(X_exog_test))