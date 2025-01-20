import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Streamlit App
st.title("Lottery Data Analysis and Modeling")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read the dataset
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Overview")
    st.write("First 5 rows of the dataset:")
    st.write(data.head())

    # Preprocessing
    st.subheader("Data Preprocessing")
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y', errors='coerce')
        data = data.dropna(subset=['Date'])
        st.write("Converted 'Date' column to datetime format and dropped invalid rows.")

    # Display summary
    st.write("Dataset Summary:")
    st.write(data.describe())

    # Visualization
    st.subheader("Data Visualization")
    column_to_plot = st.selectbox("Select a column to visualize", options=data.columns)
    if column_to_plot:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column_to_plot], kde=True)
        st.pyplot(plt)

    # Feature Selection and Modeling
    st.subheader("Model Training")

    # Select features and target
    features = st.multiselect("Select feature columns", options=data.columns)
    target = st.selectbox("Select target column", options=data.columns)

    if features and target:
        X = data[features]
        y = data[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model selection
        model_type = st.selectbox("Select model", ["Linear Regression", "Decision Tree", "Random Forest"])

        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Decision Tree":
            model = DecisionTreeRegressor()
        elif model_type == "Random Forest":
            model = RandomForestRegressor()

        # Train the model
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.write(f"Model: {model_type}")
        st.write(f"Mean Squared Error: {mse:.2f}")

        # Plot predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        st.pyplot(plt)
