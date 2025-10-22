"""
Flask Web Application for Concrete Strength Prediction
"""
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import sys
from train_mix_optimizer import generate_multiple_options

app = Flask(__name__)

# Default material prices (₹ per ton) - User can override these
DEFAULT_PRICES = {
    'cement': 8000,
    'slag': 3000,
    'fly_ash': 2500,
    'aggregates': 1500
}

# Load the trained model, scaler, and metadata
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'concrete_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'model', 'feature_names.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'model', 'metadata.pkl')

try:
    print(f"🔍 Loading model from: {BASE_DIR}")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    metadata = joblib.load(METADATA_PATH)
    print(f"✅ Model loaded successfully!")
    print(f"   Model: {metadata['model_name']}")
    print(f"   R² Score: {metadata['r2_score']:.4f}")
    print(f"   MAE: {metadata['mae']:.2f} MPa")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    print(traceback.format_exc())
    # Set all variables to None if loading fails
    model = None
    scaler = None
    feature_names = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 
                     'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age']
    metadata = {
        'model_name': 'Model Loading Failed',
        'r2_score': 0.0,
        'mae': 0.0
    }

@app.route('/health')
def health():
    """
    Health check endpoint to verify model loading
    """
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_names': feature_names,
        'metadata': metadata,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    })

@app.route('/')
def index():
    """
    Landing page / prediction form
    """
    return render_template('index.html',
                         features=feature_names,
                         r2_score=metadata['r2_score'],
                         mae=metadata['mae'],
                         model_name=metadata['model_name'])

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make concrete strength prediction
    """
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Get input data
        data = request.get_json()
        
        # Extract features in correct order
        features = []
        for feature_name in feature_names:
            value = data.get(feature_name, 0)
            features.append(float(value))
        
        # Convert to pandas DataFrame with feature names to avoid warning
        import pandas as pd
        features_df = pd.DataFrame([features], columns=feature_names)
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Predict
        predicted_strength = model.predict(features_scaled)[0]
        
        # Predict strength at different ages for the chart
        age_days = [1, 3, 7, 14, 28, 56, 90, 180, 365]
        strength_over_time = []
        
        for age in age_days:
            # Create a copy of features with different age
            features_copy = features.copy()
            features_copy[7] = age  # Age is the 8th feature (index 7)
            
            # Create DataFrame and predict
            features_age_df = pd.DataFrame([features_copy], columns=feature_names)
            features_age_scaled = scaler.transform(features_age_df)
            strength_at_age = model.predict(features_age_scaled)[0]
            strength_over_time.append(float(round(strength_at_age, 2)))
        
        # Calculate derived metrics
        cement = features[0]
        water = features[3]
        water_cement_ratio = water / cement if cement > 0 else 0
        
        # Aggregate ratio
        coarse_agg = features[5]
        fine_agg = features[6]
        total_agg = coarse_agg + fine_agg
        aggregate_cement_ratio = total_agg / cement if cement > 0 else 0
        
        # Get material prices from user input (or use defaults)
        price_cement = float(data.get('price_cement', DEFAULT_PRICES['cement']))
        price_slag = float(data.get('price_slag', DEFAULT_PRICES['slag']))
        price_fly_ash = float(data.get('price_fly_ash', DEFAULT_PRICES['fly_ash']))
        price_aggregates = float(data.get('price_aggregates', DEFAULT_PRICES['aggregates']))
        
        # Cost estimation in INR using user-provided rates
        cement_cost_inr = (cement / 1000) * price_cement
        slag_cost_inr = (features[1] / 1000) * price_slag
        fly_ash_cost_inr = (features[2] / 1000) * price_fly_ash
        aggregate_cost_inr = (total_agg / 1000) * price_aggregates
        total_cost_inr = cement_cost_inr + slag_cost_inr + fly_ash_cost_inr + aggregate_cost_inr
        
        # CO2 estimation (cement production = 0.9 kg CO2 per kg cement)
        cement_co2 = cement * 0.9
        slag_co2 = features[1] * 0.1  # Much lower for slag
        fly_ash_co2 = features[2] * 0.05  # Very low for fly ash
        total_co2 = cement_co2 + slag_co2 + fly_ash_co2
        
        # Determine strength rating
        if predicted_strength < 20:
            rating = 1
            rating_text = "WEAK"
        elif predicted_strength < 30:
            rating = 2
            rating_text = "FAIR"
        elif predicted_strength < 40:
            rating = 3
            rating_text = "GOOD"
        elif predicted_strength < 55:
            rating = 4
            rating_text = "VERY GOOD"
        else:
            rating = 5
            rating_text = "EXCELLENT"
        
        # Determine suitability (convert to bool explicitly for JSON)
        suitability = {
            'residential': bool(predicted_strength >= 20),
            'commercial': bool(predicted_strength >= 40),
            'high_rise': bool(predicted_strength >= 50),
            'bridges': bool(predicted_strength >= 60),
            'pavements': bool(25 <= predicted_strength <= 45),
            'foundations': bool(predicted_strength >= 30)
        }
        
        # Water/cement ratio assessment
        if water_cement_ratio < 0.4:
            wc_status = "Low (may be difficult to work with)"
            wc_color = "orange"
        elif water_cement_ratio <= 0.6:
            wc_status = "Optimal (good strength and workability)"
            wc_color = "green"
        else:
            wc_status = "High (may reduce durability)"
            wc_color = "red"
        
        return jsonify({
            'predicted_strength': float(round(predicted_strength, 2)),
            'rating': int(rating),
            'rating_text': str(rating_text),
            'water_cement_ratio': float(round(water_cement_ratio, 3)),
            'wc_status': str(wc_status),
            'wc_color': str(wc_color),
            'aggregate_cement_ratio': float(round(aggregate_cement_ratio, 2)),
            'cost_per_m3_inr': float(round(total_cost_inr, 2)),
            'co2_kg_per_m3': float(round(total_co2, 1)),
            'strength_chart': {
                'ages': age_days,
                'strengths': strength_over_time
            },
            'material_prices': {
                'cement': float(price_cement),
                'slag': float(price_slag),
                'fly_ash': float(price_fly_ash),
                'aggregates': float(price_aggregates)
            },
            'suitability': suitability
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in prediction: {error_details}")
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/optimize_mix', methods=['POST'])
def optimize_mix():
    """
    Reverse prediction: Given target strength, suggest optimal mix designs
    """
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Get input data
        data = request.get_json()
        
        target_strength = float(data.get('target_strength', 40))
        age = float(data.get('age', 28))
        
        # Get material prices from user input (or use defaults)
        price_cement = float(data.get('price_cement', DEFAULT_PRICES['cement']))
        price_slag = float(data.get('price_slag', DEFAULT_PRICES['slag']))
        price_fly_ash = float(data.get('price_fly_ash', DEFAULT_PRICES['fly_ash']))
        price_aggregates = float(data.get('price_aggregates', DEFAULT_PRICES['aggregates']))
        
        # Generate mix options
        options = generate_multiple_options(
            target_strength, age, model, scaler, feature_names,
            price_cement, price_slag, price_fly_ash, price_aggregates
        )
        
        # Format response
        formatted_options = []
        for option in options:
            formatted_options.append({
                'name': str(option['option_name']),
                'description': str(option['option_description']),
                'predicted_strength': float(round(option['predicted_strength'], 2)),
                'cost': float(round(option['cost'], 2)),
                'cement': float(round(option['cement'], 1)),
                'water': float(round(option['water'], 1)),
                'slag': float(round(option['blast_furnace_slag'], 1)),
                'fly_ash': float(round(option['fly_ash'], 1)),
                'coarse_aggregate': float(round(option['coarse_aggregate'], 1)),
                'fine_aggregate': float(round(option['fine_aggregate'], 1)),
                'superplasticizer': float(round(option['superplasticizer'], 1)),
                'wc_ratio': float(round(option['water'] / option['cement'], 3))
            })
        
        return jsonify({
            'target_strength': float(target_strength),
            'age': float(age),
            'options': formatted_options
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in mix optimization: {error_details}")
        return jsonify({
            'error': f'Mix optimization error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

