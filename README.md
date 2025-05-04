# AI-DemandForecast

Demand prediction and exploratory analysis for inventory optimization in small businesses.

This project aims to help small businesses, especially informal or family-owned ones, make better purchasing decisions through demand prediction models. The goal is to reduce excess or shortage of stock, improve profitability, and decrease product waste.

---

## 🎯 Project Purpose

In small businesses in Colombia (neighborhood stores), inventory management is done intuitively and without data. This project emerges as a data-based solution to:
- Offer simple but actionable demand predictions.
- Explore historical consumption patterns and seasonality.
- Empower small merchants with accessible AI tools.

---

## 📊 What does it do so far?

- Trains demand prediction models with **XGBoost**.
- Allows exploratory analysis with identification of seasonality, outliers, and trends (available in notebooks).

---

## ⚙️ Technologies used

- Python
- Pandas
- Streamlit
- XGBoost

---

## 🔨 Project Status

- ✅ Initial XGBoost model
- ✅ Complete EDA with trend analysis, seasonality, and outliers

---
## 📁 Project Structure

- `1_interface_demand_forecast.py`: Main file containing the Streamlit interface for the application.
- `models/utility_funcs.py`: Contains helper functions and the XGBoost model implementation.

## 🚀 How to Run the Application

1. Create a virtual environment:
   - Windows: `python -m venv ai-forecast`
   - Linux/Mac: `python3 -m venv ai-forecast`

2. Activate the virtual environment:
   - Windows: `ai-forecast\Scripts\activate`
   - Linux/Mac: `source ai-forecast/bin/activate`

3. Install dependencies from `requirements.txt`
4. Run the application with Streamlit: `streamlit run 1_interface_demand_forecast.py`
5. The application will automatically open in your default browser. If not, navigate to the URL shown in the terminal (typically http://localhost:8501).
---

## ⏭️ Next Steps

- [ ] Compare performance with different models
- [ ] Connect FastAPI with an interface for non-technical users
- [ ] Validate the solution in real businesses
- [ ] Add more features to a more robust MVP
---

## 🤝 Contribute or give feedback
This project is open to collaboration, validation, and improvement. If you're interested in contributing to modeling, interface design, or real-world testing, contact me:

📧 juankruizo10@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/juan-camilo-ruiz-ortiz/)  
🐙 [GitHub](https://github.com/juanksRuiz)
