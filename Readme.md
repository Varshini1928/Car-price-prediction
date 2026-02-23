#  AutoValue — Car Price Prediction

An AI-powered car price prediction app with a Python/Flask ML backend and a polished HTML/CSS/JS frontend.

---

##  Project Structure

```
car-price-prediction/
├── backend/
│   ├── app.py            # Flask REST API
│   ├── model.py          # ML model (Gradient Boosting)
│   └── requirements.txt  # Python dependencies
└── frontend/
    └── index.html        # Single-page UI
```

---

##  Backend Setup

### Prerequisites
- Python 3.9+
- pip

### Installation & Run

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Start the Flask server
python app.py
```

The API will start at **http://localhost:5000**

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/brands` | List of supported car brands |
| GET | `/api/stats` | Model statistics |
| POST | `/api/predict` | Predict car price |

### Predict Request Body

```json
{
  "brand": "Toyota",
  "year": 2020,
  "mileage": 45000,
  "engineSize": 2.0,
  "fuelType": "Gasoline",
  "transmission": "Automatic"
}
```

### Predict Response

```json
{
  "predictedPrice": 18400,
  "priceRange": { "low": 16900, "high": 19900 },
  "confidence": 87.4
}
```

---

##  Frontend Setup

No build step needed — just open the file:

```bash
open frontend/index.html
# or
# drag index.html into your browser
```

> **Note:** The frontend has a built-in local fallback predictor, so it works even without the backend running. For full accuracy, start the backend first.

---

##  ML Model Details

- **Algorithm:** Gradient Boosting Regressor (scikit-learn)
- **Trees:** 200 estimators, depth 5
- **Training data:** 5,000 synthetic records based on real market patterns
- **Features:** Year, Mileage, Engine Size, Brand, Fuel Type, Transmission
- **R² Score:** ~0.95

### Supported Brands
Toyota, Honda, Ford, BMW, Mercedes-Benz, Audi, Volkswagen, Hyundai, Kia, Nissan, Chevrolet, Tesla, Lexus, Porsche, Mazda

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask, scikit-learn, NumPy |
| Frontend | Vanilla HTML/CSS/JS |
| ML Model | Gradient Boosting Regressor |
| Fonts | Syne + DM Sans (Google Fonts) |
