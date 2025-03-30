import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset from GitHub repository
github_url = "https://raw.githubusercontent.com/Lolzzz57489034/Airport-footfall-predictor/main/Airport_Flight_Data_Final_Updated.csv"
df = pd.read_csv(github_url)

# Ensure 'Date' is in datetime format
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year.fillna(df["Date"].dt.year.mode()[0]).astype(int)

# Define airport names
airport_names = [
    "Rajiv Gandhi International Airport",
    "Chhatrapati Shivaji Maharaj International Airport",
    "Indira Gandhi International Airport",
    "Kempegowda International Airport",
    "Sardar Vallabhbhai Patel International Airport",
    "Chennai International Airport"
]

# Encode categorical features
label_encoders = {}
categorical_cols = ["Season", "Weather_Good", "Economic_Trend", "Airport"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Create a synthetic revenue column (assumption: Revenue = Footfall * Avg Spending per Passenger)
df["Revenue"] = df["Actual_Footfall"] * np.random.uniform(20, 50, size=len(df))

# Prepare dataset for ML
X = df[["Year", "Airport", "Season", "Weather_Good", "Economic_Trend", "Total_Flights", "Actual_Footfall"]]
y = df["Revenue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Airport Revenue Predictor", layout="wide")
st.title("\U0001F4B0 Airport Revenue Prediction")

# Dropdown for airport selection
selected_airport = st.selectbox("Select Airport:", airport_names)

# Map selected airport to encoded value
if selected_airport in label_encoders["Airport"].classes_:
    selected_airport_encoded = label_encoders["Airport"].transform([selected_airport])[0]
else:
    st.error("Selected airport not found in dataset!")
    st.stop()

# Dropdown for season selection
selected_season = st.selectbox("Select Season:", ["Monsoon", "Summer", "Winter"])
selected_season_encoded = label_encoders["Season"].transform([selected_season])[0]

# Select future year
future_year = st.slider("Select Future Year:", min_value=df["Year"].max() + 1, max_value=df["Year"].max() + 10, step=1)

# Input for projected flight volume and footfall
projected_flights = st.number_input("Projected Flights:", min_value=100, max_value=5000, step=50, value=1000)
projected_footfall = st.number_input("Projected Footfall:", min_value=1000, max_value=1000000, step=5000, value=50000)

# Predict button
if st.button("\U0001F680 Predict"):
    # Prepare input for prediction
    input_data = np.array([[future_year, selected_airport_encoded, selected_season_encoded, 1, 1, projected_flights, projected_footfall]])

    # Predict Revenue
    predicted_revenue = model.predict(input_data)[0]

    # Display Prediction
    st.subheader(f"\U0001F4CA Predicted Revenue: **${predicted_revenue:,.2f}**")

    # Visualization
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=df["Year"], y=df["Revenue"], marker="o", label="Past Revenue")
    plt.axvline(x=future_year, color="r", linestyle="--", label="Prediction Point")
    plt.scatter(future_year, predicted_revenue, color="red", s=100, label="Predicted Revenue")
    plt.xlabel("Year")
    plt.ylabel("Revenue ($)")
    plt.legend()
    st.pyplot(plt)
