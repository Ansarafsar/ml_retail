import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
@st.cache_resource
def load_models():
    models = {
        "Linear Regression": pickle.load(open("/workspaces/ml_retail/models/linear_regression_model.pkl", "rb")),
        "XGBoost": pickle.load(open("/workspaces/ml_retail/models/xgboost_model.pkl", "rb")),
        "ARIMA": pickle.load(open("/workspaces/ml_retail/models/arima_model.pkl", "rb")),
        "LSTM": load_model("/workspaces/ml_retail/models/lstm_model.h5", compile=False)
    }
    return models

# Load the processed dataset
@st.cache_data
def load_data():
    data = pd.read_csv("/workspaces/ml_retail/data/pp_new_data.csv")  # Replace with your processed dataset path
    return data

models = load_models()
data = load_data()

# Custom CSS for styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .stSlider>div>div>div>div {
        background-color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("📊 Retail Sales Prediction Dashboard")
st.markdown("""
Welcome to the Retail Sales Prediction Dashboard!  
This app allows you to predict sales based on various input features.  
Adjust the sliders on the left to see how the predictions change.
""")

# Model selection
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))

# Sidebar for input features
st.sidebar.header("User Input Features")
st.sidebar.subheader("Adjust the sliders below:")
quantity = st.sidebar.slider("Quantity", min_value=1, max_value=100, value=10)
discount = st.sidebar.slider("Discount", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
fulfillment_time = st.sidebar.slider("Fulfillment Time (days)", min_value=1, max_value=30, value=5)

# Additional inputs for ARIMA and LSTM
if model_choice == "ARIMA":
    months_to_forecast = st.sidebar.slider("Months to Forecast", min_value=1, max_value=12, value=1)
elif model_choice == "LSTM":
    timesteps = 3  # As per your LSTM training setup
    st.sidebar.write(f"LSTM requires {timesteps} previous time steps for prediction.")

# Function to find the closest matching row in the dataset
def find_closest_row(data, quantity, discount, fulfillment_time):
    # Calculate the Euclidean distance between the input and each row in the dataset
    data['distance'] = np.sqrt(
        (data['Quantity'] - quantity) ** 2 +
        (data['Discount'] - discount) ** 2 +
        (data['Fulfillment Time'] - fulfillment_time) ** 2
    )
    # Return the row with the smallest distance
    return data.loc[data['distance'].idxmin()]

# Preprocessing function for Linear Regression, XGBoost
def preprocess_input(row):
    # Extract the required features
    input_features = row[['Quantity', 'Discount', 'Fulfillment Time', 'Customer Age', 'CLV', 'Discount Impact', 'Product Popularity', 'Year', 'Postal Code']].values.reshape(1, -1)
    # Scale the input features (as done during training)
    scaler = StandardScaler()
    scaler.fit(data[['Quantity', 'Discount', 'Fulfillment Time', 'Customer Age', 'CLV', 'Discount Impact', 'Product Popularity', 'Year', 'Postal Code']])
    return scaler.transform(input_features)

# Preprocessing function for LSTM
def preprocess_lstm_input(quantity, discount, fulfillment_time, timesteps=3):
    input_features = np.array([[quantity, discount, fulfillment_time]])
    scaler = MinMaxScaler()
    scaler.fit([[1, 0.0, 1]])  # Dummy fit to match training scaler
    scaled_input = scaler.transform(input_features)
    return scaled_input.reshape((1, timesteps, 1))

# Model Prediction
try:
    if model_choice in ["Linear Regression", "XGBoost"]:
        # Find the closest matching row in the dataset
        closest_row = find_closest_row(data, quantity, discount, fulfillment_time)
        # Preprocess the input features
        input_features = preprocess_input(closest_row)
        # Make prediction
        prediction = models[model_choice].predict(input_features)[0]
    elif model_choice == "ARIMA":
        prediction = models[model_choice].forecast(steps=months_to_forecast)[-1]  # Last value in forecast
    elif model_choice == "LSTM":
        input_lstm = preprocess_lstm_input(quantity, discount, fulfillment_time)
        prediction_scaled = models[model_choice].predict(input_lstm)[0][0]
        scaler = MinMaxScaler()
        scaler.fit([[1]])  # Dummy fit to match training scaler
        prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]

    # Display prediction
    st.subheader("Predicted Sales")
    st.metric("Predicted Sales", f"${prediction:.2f}")

    # Visualization
    st.subheader("Sales Prediction Visualization")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Predicted Sales"], [prediction], color="#FF4B4B")  # Use a custom color
    ax.set_ylabel("Sales ($)")
    ax.set_title("Predicted Sales Value", fontsize=14)
    st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {e}")

# Debugging: Print Linear Regression coefficients
if model_choice == "Linear Regression":
    st.subheader("Linear Regression Coefficients")
    st.write(models["Linear Regression"].coef_)

# Debugging: Plot correlation heatmap
st.subheader("Feature Correlation Heatmap")

# Select only numeric columns for correlation heatmap
numeric_data = data.select_dtypes(include=['number'])
corr = numeric_data.corr()

# Plot the heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("© AK Retail Sales Prediction Dashboard. Built with ❤️ using Streamlit.")