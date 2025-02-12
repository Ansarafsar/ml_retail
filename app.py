# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns

# Import utility functions
from utils.data_loader import load_data

# Load the cleaned and processed dataset
new_data = load_data()

# Load trained models
@st.cache_resource
def load_models():
    with open('models/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/linear_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('models/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    lstm_model = load_model('models/lstm_model.h5')
    with open('models/arima_model.pkl', 'rb') as f:
        arima_model = pickle.load(f)
    return rf_model, lr_model, xgb_model, lstm_model, arima_model

rf_model, lr_model, xgb_model, lstm_model, arima_model = load_models()

# Title and layout
st.title("Retail Sales Dashboard")
st.markdown("""
This dashboard provides **data analytics** from trained models and **business intelligence** from open-source AI models.
""")

# Sidebar for user inputs
st.sidebar.header("User Inputs")
selected_model = st.sidebar.selectbox(
    "Select Model", 
    ["Linear Regression", "Random Forest", "XGBoost", "LSTM", "ARIMA"]
)
discount = st.sidebar.slider("Discount (%)", 0, 100, 10)

# Data Analytics Section
st.header("📊 Data Analytics")
st.write("### Sales Forecasting")

if selected_model == "Linear Regression":
    features = [discount]  # Replace with actual feature inputs
    prediction = lr_model.predict([features])[0]
    st.write(f"Predicted Sales (Linear Regression): {prediction:.2f}")

elif selected_model == "Random Forest":
    features = [discount]  # Replace with actual feature inputs
    prediction = rf_model.predict([features])[0]
    st.write(f"Predicted Sales (Random Forest): {prediction:.2f}")

elif selected_model == "XGBoost":
    features = [discount]  # Replace with actual feature inputs
    prediction = xgb_model.predict([features])[0]
    st.write(f"Predicted Sales (XGBoost): {prediction:.2f}")

elif selected_model == "LSTM":
    # Placeholder for LSTM input data
    st.write("LSTM predictions will appear here.")

elif selected_model == "ARIMA":
    forecast = arima_model.forecast(steps=1)[0]
    st.write(f"Predicted Sales (ARIMA): {forecast:.2f}")

# Visualizations
st.write("### Sales Trends Over Time")
plt.figure(figsize=(10, 6))
sns.lineplot(x='Sales Date', y='Sales', data=new_data)
plt.title("Sales Over Time")
st.pyplot(plt)

# Business Intelligence Section
st.header("📈 Business Intelligence")
st.write("### Optimal Discount Strategy")

profit_impact = new_data.groupby('Discount')['Profit'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='Discount', y='Profit', data=profit_impact)
plt.title("Impact of Discount on Profit")
st.pyplot(plt)

st.write(f"Optimal Discount for Maximum Profit: {profit_impact.loc[profit_impact['Profit'].idxmax(), 'Discount']}%")

# Inventory Management Insights
st.write("### Inventory Management")
lead_time = 7  # days
daily_sales = new_data.groupby('Sales Date')['Sales'].sum().mean()
safety_stock = daily_sales * lead_time
st.write(f"Safety Stock (7-day lead time): {safety_stock:.2f} units")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")