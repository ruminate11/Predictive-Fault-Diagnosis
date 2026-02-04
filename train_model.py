# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("data/motor_data.csv")

X = data.drop("fault", axis=1)
y = data["fault"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", round(accuracy*100, 2), "%")

# ✅ Save model
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")
