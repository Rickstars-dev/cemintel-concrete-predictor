# ConcretePredict AI - Concrete Strength Calculator

## 🎯 Project Overview

An AI-powered web application that predicts concrete compressive strength with **92.6% accuracy** (R² = 0.926).

Built with Machine Learning to help construction professionals optimize concrete mixes for safety and cost efficiency.

## ✨ Features

- 🤖 **Advanced ML Model** - Gradient Boosting with 92.6% accuracy
- 📊 **Real-time Predictions** - Instant strength calculations
- 💰 **Cost Estimation** - Per m³ cost analysis
- 🌍 **CO₂ Calculator** - Environmental impact assessment
- ✅ **Suitability Checker** - Application recommendations
- 📈 **Performance Metrics** - Water/cement ratio analysis

## 🏗️ Technology Stack

- **Backend:** Python 3.12, Flask
- **ML:** scikit-learn (Gradient Boosting Regressor)
- **Data:** UCI Concrete Dataset (1,030 samples)
- **Frontend:** HTML5, CSS3, JavaScript

## 📊 Model Performance

- **R² Score:** 0.926 (92.6%)
- **MAE:** ±2.89 MPa
- **RMSE:** 4.36 MPa
- **CV R² (5-fold):** 0.922

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/concrete-strength-predictor.git
cd concrete-strength-predictor

# Install dependencies
pip install -r requirements.txt

# Download dataset and train model
python download_data.py
python train_model.py

# Run application
python app.py
```

Visit: `http://127.0.0.1:5000`

## 📝 Input Features

1. **Cement** (kg/m³) - Portland cement content
2. **Blast Furnace Slag** (kg/m³) - Steel production byproduct
3. **Fly Ash** (kg/m³) - Coal combustion byproduct
4. **Water** (kg/m³) - Water content
5. **Superplasticizer** (kg/m³) - Flow improver
6. **Coarse Aggregate** (kg/m³) - Gravel/crushed stone
7. **Fine Aggregate** (kg/m³) - Sand
8. **Age** (days) - Curing time

## 🎯 Use Cases

- 🏠 **Construction Planning** - Determine optimal mix design
- 💰 **Cost Optimization** - Balance strength and cost
- 🌱 **Sustainability** - Reduce CO₂ with slag/fly ash
- 🔍 **Quality Control** - Predict strength before testing
- 📅 **Timeline Planning** - Know when concrete is ready

## 📈 Prediction Examples

| Mix Type | Strength | Applications |
|----------|----------|--------------|
| Budget Residential | 28 MPa | Houses, garages |
| Standard Commercial | 40 MPa | Offices, retail |
| High-Performance | 65 MPa | Bridges, high-rises |

## 🌍 Environmental Impact

The app calculates CO₂ emissions per m³ and shows savings when using:
- **Blast Furnace Slag** - Reduces cement by up to 50%
- **Fly Ash** - Lowers carbon footprint by 20-30%

## 📊 Dataset

- **Source:** UCI Machine Learning Repository
- **Samples:** 1,030 concrete mixes
- **Features:** 8 ingredients + age
- **Target:** Compressive strength (2.33 - 82.60 MPa)
- **Quality:** No missing values, clean data

## 🔮 How It Works

1. User enters concrete mix proportions
2. Features are scaled using StandardScaler
3. Gradient Boosting model predicts strength
4. Results include:
   - Compressive strength (MPa)
   - Quality rating (⭐⭐⭐⭐⭐)
   - Water/cement ratio analysis
   - Cost estimate
   - CO₂ emissions
   - Suitability for different applications

## 🏆 Why This Project?

- ✅ **Safety-Critical** - Concrete strength determines building safety
- ✅ **Cost Impact** - Optimize $billions in construction materials
- ✅ **Environmental** - Reduce cement (8% of global CO₂ emissions)
- ✅ **High Accuracy** - 92.6% R² score with clean physics-based features
- ✅ **Practical** - Real-world application for construction industry

## 📚 References

- **Dataset:** [UCI Concrete Data](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)
- **Paper:** Yeh, I-Cheng. "Modeling of strength of high-performance concrete using artificial neural networks." Cement and Concrete research 28.12 (1998): 1797-1808.

## 👨‍💻 Author

**Abhishek Chandra**
- Achieving 92.6% accuracy on concrete strength prediction

## 📄 License

MIT License - Feel free to use this project for learning and commercial purposes.

## 🚀 Deployment

This app can be deployed on:
- Render (recommended)
- PythonAnywhere
- Railway
- Heroku


