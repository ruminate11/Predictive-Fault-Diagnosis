# 🔧 Predictive Fault Diagnosis using Machine Learning

## 📌 Problem
Manufacturing plants face frequent **minor stoppages (< 1 min)** due to motor faults, causing significant OEE loss.

This project predicts faults early using sensor data.

---

## 🎯 Objective
Classify motor condition:
- 0 → Normal
- 1 → Fault

---

## 📊 Sensors Used
- Current
- Vibration
- Temperature

---

## 🧠 ML Model
Random Forest Classifier

Why?
- Handles noisy sensor data
- Good accuracy
- Provides feature importance

---

## ⚙️ Steps
1. Generate synthetic sensor dataset
2. Train Random Forest
3. Evaluate accuracy
4. Visualize:
   - Confusion matrix
   - Feature importance

---

## 🚀 How to Run

### Install dependencies
pip install -r requirements.txt

### Generate dataset
python generate_data.py

### Train model
python train_model.py

---

## 📈 Output
- Accuracy report
- Confusion matrix
- Feature importance graph

---

## 🏭 Business Impact (Godrej Use Case)
The model detects:
- Vibration spikes
- Temperature rise
- Current overload

These are early indicators of **minor stoppages in appliance assembly motors**, helping reduce downtime and maintenance costs.

---

## ✅ Result
Achieved ~90–95% classification accuracy.
Vibration found to be the strongest predictor of failure.
