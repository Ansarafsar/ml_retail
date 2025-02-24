import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load models and scalers
@st.cache_resource
def load_models_and_scalers():
    models = {
        "Linear Regression": pickle.load(open("/workspaces/ml_retail/models/linear_regression_model.pkl", "rb")),
        "XGBoost": pickle.load(open("/workspaces/ml_retail/models/xgboost_model.pkl", "rb")),
        "ARIMA": pickle.load(open("/workspaces/ml_retail/models/arima_model.pkl", "rb")),
        "LSTM": load_model("/workspaces/ml_retail/models/lstm_model.h5", compile=False)
    }
    scalers = {
        "Linear Regression": pickle.load(open("/workspaces/ml_retail/models/scaler.pkl", "rb")),
        "XGBoost": pickle.load(open("/workspaces/ml_retail/models/scaler_xgboost.pkl", "rb"))
    }
    return models, scalers

# Load the processed dataset
@st.cache_data
def load_data():
    data = pd.read_csv("/workspaces/ml_retail/data/pp_new_data.csv")  # Replace with your processed dataset path
    return data


models, scalers = load_models_and_scalers()
data = load_data()

# Custom CSS for styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #6aebe8;
        color: white;
        font-weight: bold;
    }
    .stSlider>div>div>div>div {
        background-color: #6aebe8;
    }
    .stDataFrame {
        font-size: 14px;
    }
    .stMetric {
        font-size: 20px;
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
quantity = int(st.sidebar.slider("Quantity", min_value=1, max_value=10, value=1))
discount = float(st.sidebar.slider("Discount", min_value=0.0, max_value=1.0, value=0.1, step=0.01))
fulfillment_time = int(st.sidebar.slider("Fulfillment Time (days)", min_value=1, max_value=10, value=1))

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
def preprocess_input(row, scaler):
    # Extract the required features
    input_features = row[['Quantity', 'Discount', 'Fulfillment Time', 'Customer Age', 'CLV', 'Discount Impact', 'Product Popularity', 'Year', 'Postal Code']].values.reshape(1, -1)
    # Scale the input features (as done during training)
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
        
        # Display the closest row in an attractive table
        st.subheader("Closest Matching Row from Dataset")
        closest_row_df = pd.DataFrame(closest_row).T  # Convert Series to DataFrame
        closest_row_df = closest_row_df.reset_index(drop=True)
        st.dataframe(closest_row_df.style.highlight_max(axis=0, color="#6aebe8"))  # Highlight max values

        # Select correct scaler based on model
        scaler = scalers[model_choice]
        # Preprocess the input features
        input_features = preprocess_input(closest_row, scaler)
        # Make prediction
        prediction = models[model_choice].predict(input_features)[0]

        # --- Visualizing Multiple Predictions ---
        # Generate a range of quantity values
        quantity_values = np.linspace(1, 10, 5)  # 5 quantity values between 1 and 10
        predictions = []
        for q in quantity_values:
            # Find the closest matching row in the dataset
            closest_row = find_closest_row(data, int(q), discount, fulfillment_time)
            # Select correct scaler based on model
            scaler = scalers[model_choice]
            # Preprocess the input features
            input_features = preprocess_input(closest_row, scaler)
            # Make prediction
            predictions.append(models[model_choice].predict(input_features)[0])

        # --- Visualization ---
        st.subheader("Sales Prediction Visualization")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(quantity_values.astype(str), predictions, color="#6aebe8")  # Custom color
        ax.set_xlabel("Quantity")
        ax.set_ylabel("Sales ($)")
        ax.set_title("Predicted Sales vs. Quantity", fontsize=14)
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels if needed

        # --- Auto-adjust y-axis scale ---
        y_min = min(predictions) * 0.9  # Reduce by 10% for padding
        y_max = max(predictions) * 1.1  # Increase by 10% for padding
        ax.set_ylim(y_min, y_max)
        st.pyplot(fig)

    elif model_choice == "ARIMA":
        prediction = models[model_choice].forecast(steps=months_to_forecast)[-1]  # Last value in forecast

        # --- Visualization for ARIMA ---
        st.subheader("ARIMA Sales Forecast")
        fig, ax = plt.subplots(figsize=(8, 4))
        forecast = models[model_choice].forecast(steps=months_to_forecast)
        ax.plot(forecast, marker='o', linestyle='-')
        ax.set_xlabel("Months")
        ax.set_ylabel("Sales")
        ax.set_title("ARIMA Sales Forecast", fontsize=14)
        st.pyplot(fig)

    elif model_choice == "LSTM":
        input_lstm = preprocess_lstm_input(quantity, discount, fulfillment_time)
        prediction_scaled = models[model_choice].predict(input_lstm)[0][0]
        scaler = MinMaxScaler()
        scaler.fit([[1]])  # Dummy fit to match training scaler
        prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]

        # --- Visualization for LSTM ---
        st.subheader("LSTM Sales Prediction")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(["Predicted Sales"], [prediction], color="#6aebe8")
        ax.set_ylabel("Sales ($)")
        ax.set_title("Predicted Sales Value (LSTM)", fontsize=14)

        # --- Auto-adjust y-axis scale ---
        y_min = prediction * 0.9  # Reduce by 10% for padding
        y_max = prediction * 1.1  # Increase by 10% for padding
        ax.set_ylim(y_min, y_max)

        st.pyplot(fig)

    # Display prediction
    st.subheader("Predicted Sales")
    st.metric("Predicted Sales", f"${prediction:.2f}")

except Exception as e:
    st.error(f"An error occurred: {e}")

# Debugging: Print Linear Regression coefficients
if model_choice == "Linear Regression":
    st.subheader("Linear Regression Coefficients")
    st.write(models["Linear Regression"].coef_)

# Debugging: Plot correlation heatmap (only for LSTM and Linear Regression/XGBoost)
if model_choice in ["Linear Regression", "XGBoost", "LSTM"]:
    st.subheader("Feature Correlation Heatmap")

    # Select only numeric columns for correlation heatmap
    numeric_data = data.select_dtypes(include=['number'])

# Debugging: Plot correlation heatmap (only for LSTM and Linear Regression/XGBoost)
if model_choice in ["Linear Regression", "XGBoost", "LSTM"]:
    st.subheader("Feature Correlation Heatmap")

    # Select only numeric columns for correlation heatmap
    numeric_data = data.select_dtypes(include=['number'])

    # Handle LSTM-specific filtering
    if model_choice == "LSTM":
        if 'Product Popularity' in numeric_data.columns:
            # Filter for top 5 products based on 'Product Popularity'
            top_products = data.nlargest(5, 'Product Popularity')['Product ID'].unique()
            lstm_data = data[data['Product ID'].isin(top_products)].select_dtypes(include=['number'])

            # Drop rows with NaN values after filtering
            lstm_data = lstm_data.dropna()

            if not lstm_data.empty:  # Check if the filtered DataFrame is not empty
                numeric_data = lstm_data
            else:
                st.warning("Warning: No data available after filtering for top products. Displaying heatmap using all numeric data.")
        else:
            st.warning("Warning: 'Product Popularity' column not found or is not numeric. Displaying heatmap using all numeric data.")

    # Plot the heatmap
    if not numeric_data.empty:
        corr = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric data available for correlation heatmap.")

# Footer
st.markdown("---")
st.markdown("© AK Retail Sales Prediction Dashboard. Built with ❤️ using Streamlit.")