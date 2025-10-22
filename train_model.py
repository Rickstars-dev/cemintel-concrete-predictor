"""
Train Concrete Strength Prediction Models
Compare multiple algorithms and select the best
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

print("=" * 70)
print("CONCRETE STRENGTH PREDICTION - MODEL TRAINING")
print("=" * 70)

# Load data
print("\n📂 Loading dataset...")
df = pd.read_csv('data/concrete_data.csv')
print(f"✅ Loaded {len(df)} samples")

# Prepare features and target
X = df.drop('compressive_strength', axis=1)
y = df['compressive_strength']

print(f"\n📊 Features: {list(X.columns)}")
print(f"   Target: compressive_strength (MPa)")
print(f"   Range: {y.min():.2f} - {y.max():.2f} MPa")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n🔀 Train/Test Split:")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Scale features
print("\n🔄 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create models directory
os.makedirs('model', exist_ok=True)

# Train models
print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),
    'Ridge Regression': Ridge(alpha=1.0)
}

results = {}

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    
    results[name] = {
        'model': model,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae': mae,
        'rmse': rmse,
        'cv_mean': cv_mean
    }
    
    print(f"   R² (Train): {r2_train:.4f}")
    print(f"   R² (Test):  {r2_test:.4f}")
    print(f"   MAE: {mae:.2f} MPa")
    print(f"   RMSE: {rmse:.2f} MPa")
    print(f"   CV R² (5-fold): {cv_mean:.4f}")

# Select best model
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'R² (Test)': [r['r2_test'] for r in results.values()],
    'MAE (MPa)': [r['mae'] for r in results.values()],
    'RMSE (MPa)': [r['rmse'] for r in results.values()],
    'CV R²': [r['cv_mean'] for r in results.values()]
})

print(f"\n{comparison.to_string(index=False)}")

best_model_name = max(results.keys(), key=lambda k: results[k]['r2_test'])
best_model = results[best_model_name]['model']
best_r2 = results[best_model_name]['r2_test']
best_mae = results[best_model_name]['mae']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R² Score: {best_r2:.4f} ({best_r2*100:.2f}%)")
print(f"   MAE: {best_mae:.2f} MPa")

if best_r2 >= 0.85:
    print(f"   ✅ TARGET ACHIEVED (R² ≥ 0.85)!")
else:
    print(f"   ⚠️  Below target, but still good performance")

# Save best model - Remove random state to avoid NumPy BitGenerator serialization issues
print(f"\n💾 Saving model files...")
import pickle

# Clean the model by removing random state attributes that cause BitGenerator errors
def clean_model_for_pickle(model):
    """Remove NumPy random state from model to avoid BitGenerator serialization issues"""
    import copy
    cleaned_model = copy.deepcopy(model)
    
    # For Gradient Boosting models
    if hasattr(cleaned_model, '_random_state'):
        cleaned_model._random_state = None
    if hasattr(cleaned_model, 'random_state'):
        # Store the seed but not the state object
        if hasattr(cleaned_model.random_state, 'bit_generator'):
            cleaned_model.random_state = 42  # Reset to seed instead of state object
    
    # For ensemble models with estimators
    if hasattr(cleaned_model, 'estimators_'):
        for estimator in cleaned_model.estimators_:
            if hasattr(estimator, '_random_state'):
                estimator._random_state = None
    
    return cleaned_model

cleaned_best_model = clean_model_for_pickle(best_model)
joblib.dump(cleaned_best_model, 'model/concrete_model.pkl', compress=3, protocol=3)
joblib.dump(scaler, 'model/scaler.pkl', compress=3, protocol=3)
joblib.dump(list(X.columns), 'model/feature_names.pkl', protocol=3)

metadata = {
    'model_name': best_model_name,
    'r2_score': best_r2,
    'mae': best_mae,
    'rmse': results[best_model_name]['rmse'],
    'samples': len(df),
    'features': list(X.columns)
}
joblib.dump(metadata, 'model/metadata.pkl', protocol=3)

print(f"   ✅ model/concrete_model.pkl")
print(f"   ✅ model/scaler.pkl")
print(f"   ✅ model/feature_names.pkl")
print(f"   ✅ model/metadata.pkl")

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print(f"\n📊 Feature Importance:")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for idx, row in feature_importance.iterrows():
        print(f"   {row['Feature']:.<25} {row['Importance']:.4f}")

print("\n" + "=" * 70)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"\n🎯 Ready to predict concrete strength with {best_r2*100:.1f}% accuracy!")
