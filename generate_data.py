# generate_data.py

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n = 3000

# Normal readings
current = np.random.normal(10, 2, n)
vibration = np.random.normal(2, 0.5, n)
temperature = np.random.normal(40, 5, n)

data = pd.DataFrame({
    "current": current,
    "vibration": vibration,
    "temperature": temperature
})

# Fault rule (engineering logic)
data["fault"] = (
    (data["vibration"] > 3) |
    (data["temperature"] > 55) |
    (data["current"] > 14)
).astype(int)

# Save file
os.makedirs("data", exist_ok=True)
data.to_csv("data/motor_data.csv", index=False)

print("Dataset generated at data/motor_data.csv")
print(data.head())
