import joblib
import numpy as np

print("Testing model loading...")
model = joblib.load('model/concrete_model.pkl')
print(f"✅ Model loaded successfully!")
print(f"Model type: {type(model).__name__}")

# Test prediction
scaler = joblib.load('model/scaler.pkl')
feature_names = joblib.load('model/feature_names.pkl')

# Sample data
sample = np.array([[300, 50, 0, 180, 5, 1000, 800, 28]])
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print(f"✅ Prediction works! Sample prediction: {prediction[0]:.2f} MPa")
