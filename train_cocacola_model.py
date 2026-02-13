import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load ONLY history data
df = pd.read_csv("Coca-Cola_stock_history.csv")

# Select basic features (furniture-style)
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save model
joblib.dump(model, "cocacola_model.pkl")

print("✅ Coca-Cola model trained using stock history data")
