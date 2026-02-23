import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Brand prestige multipliers
BRAND_ENCODING = {
    "Toyota": 1.0, "Honda": 0.95, "Ford": 0.85, "BMW": 2.2,
    "Mercedes-Benz": 2.4, "Audi": 2.0, "Volkswagen": 1.1,
    "Hyundai": 0.75, "Kia": 0.70, "Nissan": 0.90,
    "Chevrolet": 0.80, "Tesla": 2.5, "Lexus": 1.9,
    "Porsche": 4.0, "Mazda": 0.95
}

FUEL_ENCODING = {
    "Gasoline": 1.0, "Diesel": 1.05, "Hybrid": 1.2,
    "Electric": 1.4, "Plug-in Hybrid": 1.3
}

TRANSMISSION_ENCODING = {
    "Automatic": 1.05, "Manual": 0.95, "CVT": 1.0, "Semi-Automatic": 1.02
}


def generate_synthetic_data(n=5000):
    """Generate realistic synthetic car data for training."""
    np.random.seed(42)
    
    brands = list(BRAND_ENCODING.keys())
    fuels = list(FUEL_ENCODING.keys())
    transmissions = list(TRANSMISSION_ENCODING.keys())
    
    years = np.random.randint(2000, 2024, n)
    mileages = np.random.exponential(60000, n).clip(0, 300000)
    engine_sizes = np.random.choice([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0], n)
    brand_names = np.random.choice(brands, n)
    fuel_names = np.random.choice(fuels, n)
    trans_names = np.random.choice(transmissions, n)
    
    brand_enc = np.array([BRAND_ENCODING[b] for b in brand_names])
    fuel_enc = np.array([FUEL_ENCODING[f] for f in fuel_names])
    trans_enc = np.array([TRANSMISSION_ENCODING[t] for t in trans_names])
    
    # Base price formula
    age = 2024 - years
    base_price = (
        15000
        * brand_enc
        * fuel_enc
        * trans_enc
        * (1 + engine_sizes * 0.15)
        * np.exp(-0.08 * age)
        * np.exp(-0.000003 * mileages)
    )
    
    noise = np.random.normal(1.0, 0.12, n)
    prices = (base_price * noise).clip(1000, 500000)
    
    X = np.column_stack([
        years, mileages, engine_sizes, brand_enc, fuel_enc, trans_enc
    ])
    
    return X, prices


def train_model():
    print("Training car price prediction model...")
    X, y = generate_synthetic_data(5000)
    
    feature_names = ['year', 'mileage', 'engineSize', 'brandScore', 'fuelScore', 'transmissionScore']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    print(f"Model trained successfully. R² score: {model.score(X_scaled, y):.4f}")
    return model, scaler, feature_names


def predict_price(model, scaler, feature_names, data):
    brand_score = BRAND_ENCODING.get(data['brand'], 1.0)
    fuel_score = FUEL_ENCODING.get(data['fuelType'], 1.0)
    trans_score = TRANSMISSION_ENCODING.get(data['transmission'], 1.0)
    
    features = np.array([[
        float(data['year']),
        float(data['mileage']),
        float(data['engineSize']),
        brand_score,
        fuel_score,
        trans_score
    ]])
    
    features_scaled = scaler.transform(features)
    price = model.predict(features_scaled)[0]
    
    return max(500, float(price))
