# ml_retail
# Retail Sales Prediction Dashboard

## Overview

This project is a Retail Sales Prediction Dashboard built using Streamlit. It allows users to predict sales based on various input features such as quantity, discount, fulfillment time, customer age, CLV (Customer Lifetime Value), discount impact, and product popularity. The dashboard supports multiple machine learning models, including Linear Regression, XGBoost, ARIMA, and LSTM, to provide accurate sales predictions.

## Features

- **Model Selection**: Choose from multiple machine learning models (Linear Regression, XGBoost, ARIMA, LSTM) to make sales predictions.
- **User Input Features**: Adjust input features using sliders in the sidebar to see how predictions change.
- **Visualizations**: Interactive visualizations including bar charts, line charts, and correlation heatmaps.
- **Closest Matching Row**: Displays the closest matching row from the dataset based on user input.
- **Predicted Sales**: Shows the predicted sales value based on the selected model and input features.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/retail-sales-prediction-dashboard.git
   cd retail-sales-prediction-dashboard
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Select a Model**: Choose a machine learning model from the dropdown menu in the sidebar.
2. **Adjust Input Features**: Use the sliders to adjust the input features such as quantity, discount, and fulfillment time.
3. **View Predictions**: The dashboard will display the predicted sales value along with visualizations and the closest matching row from the dataset.

## Models and Data

### Models

- **Linear Regression**: Trained using scikit-learn and saved as `linear_regression_model.pkl`.
- **XGBoost**: Trained using XGBoost and saved as `xgboost_model.pkl`.
- **ARIMA**: Trained using statsmodels and saved as `arima_model.pkl`.
- **LSTM**: Trained using TensorFlow/Keras and saved as `lstm_model.h5`.

### Scalers

- **Standard Scaler**: Used for Linear Regression and XGBoost, saved as `scaler.pkl` and `scaler_xgboost.pkl`.

### Data

- **Processed Dataset**: The dataset used for training and prediction is stored as `pp_new_data.csv`. It includes features like customer age, CLV, fulfillment time, discount impact, and product popularity.

## Data Preprocessing

The dataset was preprocessed and split into features (X) and target (y) as follows:

```python
# Define features (X) and target (y)
X = new_data.drop(['Sales', 'Profit'], axis=1)
y = new_data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Visualizations

- **Bar Chart**: Shows predicted sales vs. quantity for Linear Regression and XGBoost.
- **Line Chart**: Displays ARIMA sales forecast over the selected number of months.
- **Correlation Heatmap**: Visualizes the correlation between numeric features in the dataset.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Streamlit for the interactive web app framework.
- Scikit-learn, XGBoost, statsmodels, and TensorFlow/Keras for machine learning models.
- Pandas, NumPy, Matplotlib, and Seaborn for data manipulation and visualization.

## Contact

For any questions or feedback, please reach out to [your email].

---

Enjoy using the Retail Sales Prediction Dashboard! 🚀
