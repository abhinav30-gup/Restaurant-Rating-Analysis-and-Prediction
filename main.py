import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("Dataset .csv")

print("Dataset Loaded Successfully")
print(df.head())

# =========================
# FEATURE ENGINEERING
# =========================

df['Has Table Booking'] = df['Has Table booking'].map({
    'Yes': 1,
    'No': 0
})

df['Has Online Delivery'] = df['Has Online delivery'].map({
    'Yes': 1,
    'No': 0
})

# =========================
# SELECT FEATURES
# =========================

features = [
    'Price range',
    'Votes',
    'Average Cost for two',
    'Has Table Booking',
    'Has Online Delivery'
]

X = df[features]

y = df['Aggregate rating']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL TRAINING
# =========================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed")

# =========================
# PREDICTIONS
# =========================

predictions = model.predict(X_test)

# =========================
# EVALUATION
# =========================

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("MAE:", mae)
print("R2 Score:", r2)

# =========================
# SAVE MODEL
# =========================

with open("models/restaurant_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel Saved Successfully")