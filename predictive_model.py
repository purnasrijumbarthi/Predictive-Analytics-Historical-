import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# STEP 1: ACQUIRE AND CLEAN REAL HISTORICAL DATA
# =====================================================================
# Fetching an open source historical climate dataset (Daily temperatures in Delhi)
data_url = "https://githubusercontent.com"

try:
    df = pd.read_csv(data_url)
    print("✅ Dataset successfully downloaded!")
except Exception as e:
    print(f"❌ Network error: {e}. Falling back to internal simulation data.")
    # Fallback structure if Github raw cannot be accessed
    dates = pd.date_range(start="1981-01-01", periods=1000, freq="D")
    df = pd.DataFrame({"Date": dates, "Temp": 10 + (np.arange(1000) * 0.005) + np.random.normal(0, 2, 1000)})

# Standardizing columns
df.columns = ["Date", "Target_Value"]
df["Date"] = pd.to_datetime(df["Date"])

# Introduce random missing values (NaN) to demonstrate real cleaning capabilities
np.random.seed(42)
nan_mask = np.random.choice([True, False], size=len(df), p=[0.02, 0.98])
df.loc[nan_mask, "Target_Value"] = np.nan
print(f"⚠️ Introduced {df['Target_Value'].isna().sum()} missing values for data cleaning demonstration.")

# DATA PREPROCESSING: Clean missing data points dynamically using linear interpolation
df["Target_Value"] = df["Target_Value"].interpolate(method="linear")
print("✅ Data cleaning complete. Missing records resolved.\n")

# =====================================================================
# STEP 2: FEATURE ENGINEERING (CHRONOLOGICAL TRANSLATION)
# =====================================================================
# Map text-based calendar dates to a continuous numerical feature array
df["Time_Index"] = np.arange(len(df))

X = df[["Time_Index"]]
y = df["Target_Value"]

# Chronological split to prevent data leakage (80% Training / 20% Testing)
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# =====================================================================
# STEP 3: MODEL TRAINING (TIME-SERIES LINEAR REGRESSION)
# =====================================================================
model = LinearRegression()
model.fit(X_train, y_train)

# Generate baseline historical trendline matching actual metrics
df["Historical_Trend"] = model.predict(X)

# =====================================================================
# STEP 4: MODEL EVALUATION (ACCURACY SCORES)
# =====================================================================
test_predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
r2 = r2_score(y_test, test_predictions)

# =====================================================================
# STEP 5: FUTURE OUT-OF-SAMPLE FORECASTING
# =====================================================================
# Projecting future trends for the next 180 periods (days)
forecast_horizon = 180
future_indices = np.arange(len(df), len(df) + forecast_horizon).reshape(-1, 1)
future_predictions = model.predict(future_indices)

# Compile future dataset records
future_dates = pd.date_range(start=df["Date"].max() + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
df_forecast = pd.DataFrame({
    "Date": future_dates,
    "Time_Index": future_indices.flatten(),
    "Forecasted_Trend": future_predictions
})

# =====================================================================
# STEP 6: DATA-DRIVEN FORECASTING VISUALIZATION
# =====================================================================
plt.figure(figsize=(12, 6))

# Plot historical actuals
plt.plot(df["Date"], df["Target_Value"], label="Cleaned Historical Actuals", color="#2c3e50", alpha=0.5, linewidth=1.5)
# Plot training fitted model line
plt.plot(df["Date"], df["Historical_Trend"], label="Model Fitted Historical Trend", color="#2980b9", linestyle="--", linewidth=2)
# Plot future predicted vector
plt.plot(df_forecast["Date"], df_forecast["Forecasted_Trend"], label=f"{forecast_horizon}-Day Future Forecast", color="#e74c3c", linewidth=3)

# Graphics adjustment
plt.title("Predictive Modeling Engine: Trend Analysis & Multi-Step Forecasting", fontsize=14, fontweight="bold")
plt.xlabel("Timeline Context", fontsize=11)
plt.ylabel("Data Metric Values", fontsize=11)
plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

# Save image file to directory for presentation/readme upload
plt.savefig("forecast_results.png", dpi=300)
print("💾 Dashboard visualization successfully exported to local storage as 'forecast_results.png'.")

# Output execution logs
print("\n" + "="*40)
print("       INTERNSHIP PERFORMANCE LOGS      ")
print("="*40)
print(f"Validation Root Mean Squared Error (RMSE) : {rmse:.4f}")
print(f"Validation R² Score (Variance Captured)   : {r2:.4f}")
print("="*40)
