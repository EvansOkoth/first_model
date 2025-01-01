import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Custom styles for a modern look
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        max-width: 1200px;
        margin: auto;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        background-color: white;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #007BFF;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    .stTitle {
        font-weight: bold;
        font-size: 36px;
        text-align: center;
        margin-bottom: 20px;
        color: black;
        visibility: visible;
    }
    .stSubheader {
        color: #0056b3;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Dataset Preview", "Model Performance", "Make a Prediction"])

# App title
st.markdown("<div class='stTitle'>🏠 House Price Prediction</div>", unsafe_allow_html=True)

# File uploader for dataset on the Home page
if page == "Home":
    st.write("Welcome to the modern house price prediction app. Upload your dataset, explore the data, and make predictions with ease.")
    df_file = st.file_uploader("📂 Upload your dataset (CSV format):", type="csv")
    if df_file is not None:
        st.session_state["uploaded_df"] = pd.read_csv(df_file)
        st.success("Dataset uploaded successfully!")

if "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"]

    if page == "Dataset Preview":
        st.subheader("📊 Dataset Preview (First 5 Rows):")
        st.dataframe(df.head())  # Display only the first 5 rows

    elif page == "Model Performance":
        if 'Y house price of unit area' in df.columns:
            X = df.drop(columns=['Y house price of unit area'])
            y = df['Y house price of unit area']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Define models
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }

            # Train models and calculate MSE
            mse_scores = {}
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_scores[model_name] = mse

            # Find the model with the lowest MSE
            best_model_name = min(mse_scores, key=mse_scores.get)

            st.subheader("📈 Model Performance:")
            st.write("Mean Squared Error for each model:")
            st.table(pd.DataFrame(mse_scores.items(), columns=["Model", "MSE"]))
            st.write(f"**Best Model Selected:** {best_model_name} with MSE = {mse_scores[best_model_name]:.4f}")

        else:
            st.error("The dataset must contain the 'Y house price of unit area' column.")

    elif page == "Make a Prediction":
        if 'Y house price of unit area' in df.columns:
            X = df.drop(columns=['Y house price of unit area'])
            y = df['Y house price of unit area']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Define models
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }

            # Train models and calculate MSE
            mse_scores = {}
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_scores[model_name] = mse

            # Find the model with the lowest MSE
            best_model_name = min(mse_scores, key=mse_scores.get)
            best_model = models[best_model_name]

            st.subheader("🛠️ Make a Prediction:")
            st.write("Enter the values for the input features below:")
            input_data = {}
            for col in X.columns:
                input_data[col] = st.number_input(f"{col}", value=0.0)

            input_df = pd.DataFrame([input_data])
            prediction = best_model.predict(input_df)[0]

            st.subheader("🔮 Prediction Result:")
            st.write(f"**Predicted House Price:** {prediction}")

        else:
            st.error("The dataset must contain the 'Y house price of unit area' column.")
