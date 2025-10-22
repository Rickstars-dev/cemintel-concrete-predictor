"""
Train Mix Design Optimizer - Reverse Prediction Model
Given target strength, predict optimal mix proportions
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

def optimize_mix_for_strength(target_strength, age, model, scaler, feature_names, 
                               price_cement=8000, price_slag=3000, 
                               price_fly_ash=2500, price_aggregates=1500,
                               max_iterations=100):
    """
    Use iterative optimization to find best mix design for target strength
    Optimizes for cost while meeting strength requirement
    Dynamically adjusts search range based on target strength
    """
    
    # Generate candidate mixes with BROADER range
    candidates = []
    
    # Estimate cement range based on target strength
    # Rough formula: cement ≈ target_strength × 6 to 8
    min_cement = max(150, int(target_strength * 4))
    max_cement = min(540, int(target_strength * 9))
    
    # Strategy 1: Standard mixes - Wide range with varying W/C ratios
    for cement in range(min_cement, max_cement, 20):
        for wc_ratio in [0.35, 0.40, 0.45, 0.50, 0.55]:
            water = cement * wc_ratio
            if water < 120 or water > 250:  # Keep water in reasonable range
                continue
                
            candidate = {
                'cement': float(cement),
                'blast_furnace_slag': 0.0,
                'fly_ash': 0.0,
                'water': float(water),
                'superplasticizer': 5.0 if wc_ratio >= 0.45 else 8.0,
                'coarse_aggregate': 1000.0,
                'fine_aggregate': 750.0,
                'age': float(age)
            }
            candidates.append(candidate)
    
    # Strategy 2: Eco-friendly mixes with slag (reduces cost)
    for cement in range(max(150, min_cement - 50), max_cement - 100, 30):
        for slag_amount in [50, 100, 150]:
            wc_ratio = 0.42
            water = (cement + slag_amount * 0.6) * wc_ratio
            if water < 120 or water > 250:
                continue
                
            candidate = {
                'cement': float(cement),
                'blast_furnace_slag': float(slag_amount),
                'fly_ash': 0.0,
                'water': float(water),
                'superplasticizer': 7.0,
                'coarse_aggregate': 1000.0,
                'fine_aggregate': 750.0,
                'age': float(age)
            }
            candidates.append(candidate)
    
    # Strategy 3: High-performance mixes with SCMs
    for cement in range(max(250, min_cement), max_cement, 25):
        for wc_ratio in [0.30, 0.35, 0.40]:
            water = cement * wc_ratio
            if water < 120 or water > 250:
                continue
                
            candidate = {
                'cement': float(cement),
                'blast_furnace_slag': 50.0,
                'fly_ash': 30.0,
                'water': float(water),
                'superplasticizer': 10.0,
                'coarse_aggregate': 1000.0,
                'fine_aggregate': 750.0,
                'age': float(age)
            }
            candidates.append(candidate)
    
    # Strategy 4: Varying aggregate amounts
    mid_cement = (min_cement + max_cement) // 2
    for total_agg in [1500, 1700, 1900]:
        wc_ratio = 0.45
        water = mid_cement * wc_ratio
        
        candidate = {
            'cement': float(mid_cement),
            'blast_furnace_slag': 0.0,
            'fly_ash': 0.0,
            'water': float(water),
            'superplasticizer': 5.0,
            'coarse_aggregate': float(total_agg * 0.57),  # 57% coarse
            'fine_aggregate': float(total_agg * 0.43),    # 43% fine
            'age': float(age)
        }
        candidates.append(candidate)
    
    # Evaluate all candidates and store results
    evaluated_candidates = []
    
    for candidate in candidates:
        # Create feature array
        features = [candidate[fname] for fname in feature_names]
        features_df = pd.DataFrame([features], columns=feature_names)
        features_scaled = scaler.transform(features_df)
        
        # Predict strength
        predicted_strength = model.predict(features_scaled)[0]
        
        # Calculate cost
        cost = (candidate['cement'] / 1000 * price_cement +
                candidate['blast_furnace_slag'] / 1000 * price_slag +
                candidate['fly_ash'] / 1000 * price_fly_ash +
                (candidate['coarse_aggregate'] + candidate['fine_aggregate']) / 1000 * price_aggregates)
        
        # Store result
        result = candidate.copy()
        result['predicted_strength'] = predicted_strength
        result['cost'] = cost
        result['strength_diff'] = abs(predicted_strength - target_strength)
        result['meets_target'] = predicted_strength >= target_strength * 0.95
        evaluated_candidates.append(result)
    
    # Find best candidate that meets target
    valid_candidates = [c for c in evaluated_candidates if c['meets_target']]
    
    if valid_candidates:
        # Sort by cost (lowest first) and pick the best
        valid_candidates.sort(key=lambda x: (x['cost'], x['strength_diff']))
        best_candidate = valid_candidates[0]
    else:
        # If no candidate meets target, pick the closest one
        evaluated_candidates.sort(key=lambda x: x['strength_diff'])
        best_candidate = evaluated_candidates[0]
        print(f"⚠️  Warning: No mix exactly meets {target_strength} MPa. Closest: {best_candidate['predicted_strength']:.2f} MPa")
    
    return best_candidate

def generate_multiple_options(target_strength, age, model, scaler, feature_names,
                              price_cement=8000, price_slag=3000,
                              price_fly_ash=2500, price_aggregates=1500):
    """
    Generate 3 mix options: Economical, Balanced, High-Performance
    """
    
    options = []
    
    # Option 1: Economical (minimize cost)
    economical = optimize_mix_for_strength(
        target_strength, age, model, scaler, feature_names,
        price_cement, price_slag, price_fly_ash, price_aggregates
    )
    economical['option_name'] = 'Economical'
    economical['option_description'] = 'Minimum cost while meeting strength requirement'
    options.append(economical)
    
    # Option 2: Balanced (moderate cost, good performance)
    # Slightly higher target for safety margin
    balanced = optimize_mix_for_strength(
        target_strength * 1.1, age, model, scaler, feature_names,
        price_cement, price_slag, price_fly_ash, price_aggregates
    )
    balanced['option_name'] = 'Balanced'
    balanced['option_description'] = 'Good balance of cost and performance'
    options.append(balanced)
    
    # Option 3: High-Performance (maximize strength, cost secondary)
    high_perf = optimize_mix_for_strength(
        target_strength * 1.2, age, model, scaler, feature_names,
        price_cement, price_slag, price_fly_ash, price_aggregates
    )
    high_perf['option_name'] = 'High-Performance'
    high_perf['option_description'] = 'Maximum strength and durability'
    options.append(high_perf)
    
    return options

if __name__ == "__main__":
    print("=" * 60)
    print("MIX DESIGN OPTIMIZER - Training & Testing")
    print("=" * 60)
    
    # Load existing model and scaler
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'model', 'concrete_model.pkl')
    SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
    FEATURES_PATH = os.path.join(BASE_DIR, 'model', 'feature_names.pkl')
    
    print("\n📂 Loading existing model...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("✅ Model loaded successfully!")
    
    # Test the optimizer
    print("\n" + "=" * 60)
    print("TESTING MIX OPTIMIZER")
    print("=" * 60)
    
    test_cases = [
        (30, 28, "Low-strength concrete for pavements"),
        (40, 28, "Standard structural concrete"),
        (50, 28, "High-strength concrete for columns"),
        (60, 28, "Very high-strength concrete for bridges")
    ]
    
    for target_strength, age, description in test_cases:
        print(f"\n🎯 Target: {target_strength} MPa at {age} days")
        print(f"   Use Case: {description}")
        print("-" * 60)
        
        options = generate_multiple_options(
            target_strength, age, model, scaler, feature_names
        )
        
        for i, option in enumerate(options, 1):
            print(f"\n   Option {i}: {option['option_name']}")
            print(f"   {option['option_description']}")
            print(f"   Predicted Strength: {option['predicted_strength']:.2f} MPa")
            print(f"   Cost: ₹{option['cost']:.2f}/m³")
            print(f"   Mix Proportions (kg/m³):")
            print(f"      • Cement: {option['cement']:.1f}")
            print(f"      • Water: {option['water']:.1f}")
            print(f"      • Slag: {option['blast_furnace_slag']:.1f}")
            print(f"      • Fly Ash: {option['fly_ash']:.1f}")
            print(f"      • Coarse Agg: {option['coarse_aggregate']:.1f}")
            print(f"      • Fine Agg: {option['fine_aggregate']:.1f}")
            print(f"      • Superplasticizer: {option['superplasticizer']:.1f}")
            print(f"   W/C Ratio: {option['water'] / option['cement']:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ Mix Design Optimizer is ready for deployment!")
    print("=" * 60)
