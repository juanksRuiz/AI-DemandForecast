# Usa la imagen oficial de Python como base
FROM python:3.11

# Configura el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de dependencias para la API
COPY ./requirements-api.txt /app/requirements.txt

# Instala las dependencias de la API
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copia todos los archivos necesarios para la ejecución de la API
COPY ./main.py /app/main.py
COPY ./model /app/model
# Copia específicamente el archivo metadata.json (por si acaso)
COPY ./model/metadata.json /app/model/metadata.json

# Comando para ejecutar la aplicación con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
