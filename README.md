# AI-DemandForecast

Predicción de demanda y análisis exploratorio para optimización de inventarios en pequeños negocios.

Este proyecto busca ayudar a negocios pequeños —especialmente informales o familiares— a tomar mejores decisiones de compra mediante modelos de predicción de demanda. El objetivo es reducir el exceso o faltante de stock, mejorar la rentabilidad y disminuir el desperdicio de productos.

---

## 🎯 Propósito del proyecto

En muchas regiones como Colombia, la gestión del inventario se hace de forma intuitiva y sin datos. Este proyecto surge como una solución basada en datos para:
- Ofrecer predicciones de demanda simples pero accionables.
- Explorar patrones históricos de consumo y estacionalidad.
- Empoderar a pequeños comerciantes con herramientas de IA accesibles.

---

## 📊 ¿Qué hace hasta ahora?

- Entrena modelos de predicción de demanda con **XGBoost** y **SARIMAX**.
- Permite análisis exploratorio con identificación de estacionalidad, outliers y tendencia (disponible en notebooks).
- Implementa una **API con FastAPI (en desarrollo)** para servir predicciones de forma programática.
- Incluye un archivo **Dockerfile** para encapsular el entorno y facilitar despliegues futuros.

---

## ⚙️ Tecnologías utilizadas

- Python (Pandas, Scikit-learn, Statsmodels, XGBoost)
- FastAPI (en desarrollo)
- Docker (configurado para facilitar el despliegue)
- Notebooks Jupyter para análisis EDA

---

## 🔨 Estado del proyecto

- ✅ EDA completo con análisis de tendencia, estacionalidad y outliers
- ✅ Modelos iniciales XGBoost y SARIMAX entrenados (en etapa de validación)
- 🚧 FastAPI en desarrollo
- 🚧 Preparación de pruebas con datos simulados y sintéticos
- 🚧 Definición de MVP para integración web ligera

---

## ⏭️ Próximos pasos

- [ ] Afinar modelos y comparar rendimiento
- [ ] Implementar prototipo web básico (Streamlit o Gradio)
- [ ] Conectar FastAPI con una interfaz para usuarios no técnicos
- [ ] Validar la solución en comercios reales
- [ ] Documentar el flujo completo de entrada/salida para facilitar contribuciones

---

## 🤝 Contribuye o da feedback

Este proyecto está abierto a colaboración, validación y mejora. Si te interesa aportar en modelado, diseño de interfaces o pruebas reales, contáctame:

📧 juankruizo10@gmail.com  
📍 Bogotá, Colombia
