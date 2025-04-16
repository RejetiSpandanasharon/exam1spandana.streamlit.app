import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder

# Displaying your Name and Banner ID
st.title("Car Price Prediction Analysis")
st.write("## Name - Rejeti Spandana Sharon")
st.write("## Banner ID - 001409139")

# --- Part 1: Car Price Prediction ---
# Data Loading Section for Part 1
st.header("Part 1: Car Price Prediction")
uploaded_file_1 = st.file_uploader("Upload a CSV file for Car Price Prediction", type=["csv"])
if uploaded_file_1 is not None:
    data_1 = pd.read_csv(uploaded_file_1)
    st.write(data_1.head())

# Data Exploration Section for Part 1
st.subheader("Data Exploration for Car Price Dataset")
if uploaded_file_1 is not None:
    st.write("Shape of the data: ", data_1.shape)
    st.write("Data Types: ", data_1.dtypes)
    st.write("Missing Values: ", data_1.isnull().sum())

    # Visualizing the distribution of the car prices
    st.subheader("Distribution of Car Prices")
    sns.histplot(data_1['price'], kde=True)
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    st.pyplot()

    # Correlation Heatmap for Part 1
    st.subheader("Correlation Heatmap for Car Price Dataset")
    corr = data_1.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot()

# Data Preprocessing for Part 1
st.header("Data Preprocessing for Car Price Dataset")
if uploaded_file_1 is not None:
    # Handling missing data
    data_1.fillna(data_1.mean(), inplace=True)

    # Converting categorical variables into dummy variables
    data_1 = pd.get_dummies(data_1, drop_first=True)

    # Displaying processed data
    st.write(data_1.head())

# Splitting Data for Model in Part 1
st.header("Train Test Split for Car Price Prediction")
if uploaded_file_1 is not None:
    X_1 = data_1.drop(columns='price')
    y_1 = data_1['price']

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=42)
    st.write("Training data shape: ", X_train_1.shape)
    st.write("Testing data shape: ", X_test_1.shape)

# Model Training and Prediction for Part 1
st.header("Linear Regression Model for Car Price Prediction")
if uploaded_file_1 is not None:
    model_1 = LinearRegression()
    model_1.fit(X_train_1, y_train_1)

    y_pred_1 = model_1.predict(X_test_1)

    # Displaying model metrics for Part 1
    st.write("Model Coefficients: ", model_1.coef_)
    st.write("Intercept: ", model_1.intercept_)

    mae_1 = mean_absolute_error(y_test_1, y_pred_1)
    mse_1 = mean_squared_error(y_test_1, y_pred_1)
    rmse_1 = np.sqrt(mse_1)
    r2_1 = r2_score(y_test_1, y_pred_1)

    st.write(f"Mean Absolute Error: {mae_1}")
    st.write(f"Mean Squared Error: {mse_1}")
    st.write(f"Root Mean Squared Error: {rmse_1}")
    st.write(f"R-squared: {r2_1}")

    # Residual Plot for Part 1
    st.subheader("Residual Plot for Car Price Model")
    sns.residplot(x=y_pred_1, y=y_test_1 - y_pred_1, lowess=True, color='blue', line_kws={'color': 'red', 'lw': 1})
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    st.pyplot()

# Model Summary for Part 1
st.header("Model Summary for Car Price Prediction")
if uploaded_file_1 is not None:
    X_train_with_const_1 = sm.add_constant(X_train_1)
    model_sm_1 = sm.OLS(y_train_1, X_train_with_const_1).fit()
    st.write(model_sm_1.summary())

# --- Part 2: Further Car Price Prediction Analysis ---
# Data Loading for Part 2
st.header("Part 2: Car Price Prediction Analysis")
uploaded_file_2 = st.file_uploader("Upload a CSV file for Further Car Price Prediction Analysis", type=["csv"])
if uploaded_file_2 is not None:
    data_2 = pd.read_csv(uploaded_file_2)
    st.write(data_2.head())

# Data Inspection for Part 2
st.subheader("Data Inspection for Car Price Dataset")
if uploaded_file_2 is not None:
    st.write("Data Types: ", data_2.dtypes)
    st.write("Missing Values: ", data_2.isnull().sum())

# Descriptive Statistics for Part 2
st.header("Descriptive Statistics for Car Price Data")
if uploaded_file_2 is not None:
    st.write(data_2.describe())

# Encoding Categorical Variables for Part 2
st.header("Encoding Categorical Variables for Car Price Data")
if uploaded_file_2 is not None:
    encoder = LabelEncoder()
    data_2['Category'] = encoder.fit_transform(data_2['Category'])
    st.write(data_2.head())

# Regression Analysis for Part 2
st.header("Regression Analysis for Car Price Dataset")
if uploaded_file_2 is not None:
    X_2 = data_2[['Horsepower', 'Category']]
    X_2 = sm.add_constant(X_2)
    y_2 = data_2['Price']

    # Running Regression Analysis
    regression_model_2 = sm.OLS(y_2, X_2).fit()
    st.write("Regression Analysis Summary for Car Price Dataset:")
    st.write(regression_model_2.summary())

# Variance Inflation Factor (VIF) for Part 2
st.header("Variance Inflation Factor (VIF) Calculation for Car Price Dataset")
if uploaded_file_2 is not None:
    vif_data_2 = pd.DataFrame()
    vif_data_2["Variable"] = X_2.columns
    vif_data_2["VIF"] = [variance_inflation_factor(X_2.values, i) for i in range(len(X_2.columns))]

    st.write(vif_data_2)
