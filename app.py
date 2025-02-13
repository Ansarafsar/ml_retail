import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.losses import MeanSquaredError

# Import utility functions
from utils.data_loader import load_data

# Load the cleaned and processed dataset
new_data = load_data()

# Load trained models
@st.cache_resource
def load_models():
    with open('/workspaces/ml_retail/models/linear_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('/workspaces/ml_retail/models/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    custom_objects = {'mse': MeanSquaredError()}
    lstm_model = load_model('/workspaces/ml_retail/models/lstm_model.h5', custom_objects=custom_objects)
    with open('/workspaces/ml_retail/models/arima_model.pkl', 'rb') as f:
        arima_model = pickle.load(f)
    return lr_model, xgb_model, lstm_model, arima_model

lr_model, xgb_model, lstm_model, arima_model = load_models()

# Title and layout
st.title("Retail Sales Dashboard")
st.markdown("""
This dashboard provides **data analytics** from trained models and **business intelligence** from open-source AI models.
""")

# Define region and category columns and mappings
region_columns = ['Region_North', 'Region_South', 'Region_West']
region_map = {
    'Region_North': 'North',
    'Region_South': 'South',
    'Region_West': 'West'
}

category_columns = [
    'Category of Goods_Electric Appliances',
    'Category of Goods_Fast Food',
    'Category of Goods_Furniture',
    'Category of Goods_Household Items',
    'Category of Goods_Sessional Fruits & Vegetables'
]
category_map = {
    'Category of Goods_Electric Appliances': 'Electric Appliances',
    'Category of Goods_Fast Food': 'Fast Food',
    'Category of Goods_Furniture': 'Furniture',
    'Category of Goods_Household Items': 'Household Items',
    'Category of Goods_Sessional Fruits & Vegetables': 'Sessional Fruits & Vegetables'
}

# Sidebar for user inputs
st.sidebar.header("User Inputs")
selected_model = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression", "XGBoost", "LSTM", "ARIMA"]
)

# Add feature inputs
discount = st.sidebar.slider("Discount (%)", 0, 100, 10)
region = st.sidebar.selectbox("Region", list(region_map.values()))
category = st.sidebar.selectbox("Category of Goods", list(category_map.values()))
quantity = st.sidebar.slider("Quantity", 1, 100, 10)

# Find the binary columns for the selected region and category
selected_region_column = [col for col, name in region_map.items() if name == region][0]
selected_category_column = [col for col, name in category_map.items() if name == category][0]

# Create the features array dynamically
features = np.array([[discount, new_data[selected_region_column].iloc[0], new_data[selected_category_column].iloc[0], quantity, 0, 0, 0, 0, 0]])

# Debug: Print the features array
st.write("Features Array:", features)

# Update predictions based on selected model
if selected_model == "Linear Regression":
    prediction = lr_model.predict(features)[0]
    st.write(f"Predicted Sales (Linear Regression): {prediction:.2f}")

elif selected_model == "XGBoost":
    prediction = xgb_model.predict(features)[0]
    st.write(f"Predicted Sales (XGBoost): {prediction:.2f}")

elif selected_model == "LSTM":
    # Preprocess input data for LSTM
    lstm_input = features.reshape((features.shape[0], 1, features.shape[1]))  # Reshape to (batch_size, timesteps, features)
    try:
        lstm_prediction = lstm_model.predict(lstm_input)[0][0]
        st.write(f"Predicted Sales (LSTM): {lstm_prediction:.2f}")
    except Exception as e:
        st.error(f"Error in LSTM prediction: {e}")

elif selected_model == "ARIMA":
    forecast = arima_model.forecast(steps=1).iloc[0]
    st.write(f"Predicted Sales (ARIMA): {forecast:.2f}")

# Filter the dataset based on user inputs
filtered_data = new_data[
    (new_data['Discount'] == discount) &
    (new_data[selected_region_column] == 1) &
    (new_data[selected_category_column] == 1)
]

# Check if filtered_data is empty
if filtered_data.empty:
    st.warning("No data matches the selected filters. Using the full dataset for visualization.")
    filtered_data = new_data  # Fallback to the full dataset

# Visualizations
st.write("### Sales Trends Over Time")
plt.figure(figsize=(10, 6))
sns.lineplot(x='Sales Date', y='Sales', data=filtered_data)
plt.title("Sales Over Time")
st.pyplot(plt)

# Impact of Discount on Profit
st.write("### Impact of Discount on Profit")
plt.figure(figsize=(10, 6))
sns.lineplot(x='Discount', y='Profit', data=filtered_data)
plt.title("Discount vs Profit")
st.pyplot(plt)

# Key Metrics
st.write("### Key Metrics")
total_sales = filtered_data['Sales'].sum()
average_profit = filtered_data['Profit'].mean()

st.metric("Total Sales", f"${total_sales:,.2f}")
st.metric("Average Profit", f"${average_profit:,.2f}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")