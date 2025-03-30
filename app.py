import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset from GitHub
github_url = "https://raw.githubusercontent.com/Lolzzz57489034/Airport-footfall-predictor/main/Airport_Flight_Data_Final_Updated.csv"
df = pd.read_csv(github_url)

# Ensure 'Date' is in datetime format
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year.fillna(df["Date"].dt.year.mode()[0]).astype(int)

# Encode categorical features
label_encoders = {}
categorical_cols = ["Season", "Weather_Good", "Economic_Trend", "Airport"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Airport Names from Data (Fix: Dynamic List Instead of Hardcoded)
airport_names = df["Airport"].unique()
airport_names_display = label_encoders["Airport"].inverse_transform(airport_names)

# Prepare dataset for ML
X = df[["Year", "Airport", "Season", "Weather_Good", "Economic_Trend", "Total_Flights"]]
y = df["Actual_Footfall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Airport Footfall Predictor", layout="wide")
st.title("\U00002708 Airport Footfall Prediction")

# Dropdown for airport selection
selected_airport = st.selectbox("Select Airport:", airport_names_display)
selected_airport_encoded = label_encoders["Airport"].transform([selected_airport])[0]

# Dropdown for season selection
selected_season = st.selectbox("Select Season:", df["Season"].unique())
selected_season_encoded = label_encoders["Season"].transform([selected_season])[0]

# Select future year
future_year = st.slider("Select Future Year:", min_value=df["Year"].max() + 1, max_value=df["Year"].max() + 10, step=1)

# Predict button
if st.button("\U0001F680 Predict"):
    # Use Mean Values Instead of Hardcoded Inputs
    avg_weather = int(df["Weather_Good"].mean())
    avg_economic_trend = int(df["Economic_Trend"].mean())
    avg_flights = int(df["Total_Flights"].median())

    # Prepare input for prediction
    input_data = np.array([[future_year, selected_airport_encoded, selected_season_encoded, avg_weather, avg_economic_trend, avg_flights]])

    # Predict Footfall
    predicted_footfall = model.predict(input_data)[0]

    # Display Prediction
    st.subheader(f"\U0001F4CA Predicted Footfall: **{int(predicted_footfall)} passengers**")

    # Visualization Fix: Aggregate Data Before Plotting
    df_grouped = df.groupby("Year")["Actual_Footfall"].sum().reset_index()

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=df_grouped["Year"], y=df_grouped["Actual_Footfall"], marker="o", label="Past Data")
    plt.axvline(x=future_year, color="r", linestyle="--", label="Prediction Point")
    plt.scatter(future_year, predicted_footfall, color="red", s=100, label="Predicted Footfall")
    plt.xlabel("Year")
    plt.ylabel("Passenger Footfall")
    plt.legend()
    st.pyplot(plt)
