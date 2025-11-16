# Movie Revenue Prediction & Hit Classification System

## 1. Problem Statement

This project addresses two key challenges in the film industry:
- **Regression Problem**: Predict movie revenue based on features like budget, popularity, runtime, and ratings
- **Classification Problem**: Classify whether a movie will be a "hit" (revenue > 2x budget)

## 2. Dataset Details

### Source
- **TMDB (The Movie Database)** with 5,000 movies
- **Files**: `5kMovies.csv`, `5kCredits.csv`

### Features Used
**Numerical Features**:
- `runtime`: Movie duration in minutes
- `popularity`: TMDB popularity score
- `vote_avg`: Average user rating
- `budget`: Production budget
- `revenue`: Total revenue

**Derived Features**:
- `log_budget`: Natural log of budget (for normalization)
- `log_revenue`: Natural log of revenue (target for regression)
- `is_hit`: Boolean (True if revenue > 2x budget, target for classification)

**Genre Features** (One-hot encoded):
- Action, Adventure, Fantasy, Science Fiction, Thriller, Comedy

## 3. Regression Model

### Target Variable
- `log_revenue` (log-transformed revenue for better model performance)

### Features Used
- `runtime`, `popularity`, `vote_avg`, `log_budget`

### Models Implemented
1. **Linear Regression** (Baseline)
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**
4. **XGBoost Regressor**

### Model Selection Process
- **Evaluation Metrics**: RMSE, R² Score, MAE
- **Validation Strategy**: Train-Validation-Test split (60-20-20)
- **Best Performing Model**: Gradient Boosting Regressor

### Final Regression Model Parameters
```python
GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    loss='squared_error'
)
```

## 4. Model Comparison & Selection

### Performance Comparison
```
                     RMSE      R²     MAE
Linear Regression  1.3267  0.6199  0.9125
Random Forest      1.2949  0.6379  0.8572
Gradient Boosting  1.2302  0.6732  0.8282
XGB Regression     1.2637  0.6552  0.8519
```

### Why Gradient Boosting Was Chosen
1. **Best Overall Performance**: Lowest RMSE and MAE, highest R²
2. **Robustness**: Handles non-linear relationships well
3. **Feature Importance**: Provides interpretable feature weights
4. **Regularization**: Built-in prevention of overfitting

### Hyperparameter Tuning Approach
- **Random Forest**: Grid search for `max_depth` and `n_estimators`
- **Gradient Boosting**: Used recommended parameters with proven performance
- **XGBoost**: Similar structure to Gradient Boosting for fair comparison

## 5. Classification Model

### Target Variable
- `is_hit`: Binary classification (True/False)

### Features Used
- `runtime`, `popularity`, `vote_avg`, `log_budget`

### Model Used
- **Logistic Regression** with L2 regularization

### Evaluation Metrics
- **AUC-ROC**: Area Under ROC Curve
- **Precision-Recall**: Trade-off analysis
- **F1-Score**: Balanced metric for imbalanced data

## 6. K-Fold Cross Validation for Optimal C

### Process Description
```python
# 5-Fold Cross Validation
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

# Tested C values
C_values = [0.001, 0.01, 0.1, 0.5, 1, 5, 10]

# Selection based on highest mean AUC score
```

### Results
```
C=0.001 0.XXX +- 0.XXXXXX
C=0.01  0.XXX +- 0.XXXXXX
C=0.1   0.XXX +- 0.XXXXXX  ← Best performing
C=0.5   0.XXX +- 0.XXXXXX
C=1     0.XXX +- 0.XXXXXX
C=5     0.XXX +- 0.XXXXXX
C=10    0.XXX +- 0.XXXXXX
```

### Optimal Parameter
- **Best C**: 0.1 (provides optimal regularization strength)
- **Rationale**: Balances bias-variance tradeoff, prevents overfitting

## 7. Converting Jupyter Notebook to Python Script

### Installation
```bash
pip install jupytext
```

### Conversion Commands
```bash
# Convert single notebook
jupytext --to py notebook.ipynb

# Convert all notebooks in directory
jupytext --to py *.ipynb

# Set percent format (recommended)
jupytext --set-formats ipynb,py:percent notebook.ipynb

# Convert back to notebook
jupytext --to notebook script.py
```

### File Structure After Conversion
```
project/
├── classification.py          # Converted from .ipynb
├── regression.py             # Converted from .ipynb
├── requirements.txt
├── Dockerfile
└── src/
    ├── models/               # Trained model binaries
    └── data/                 # Dataset files
```

## 8. Requirements & Installation

### PIPFile
```ini
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
pandas = ">=1.5.0"
numpy = ">=1.21.0"
scikit-learn = ">=1.0.0"
xgboost = ">=1.5.0"
matplotlib = ">=3.5.0"
seaborn = ">=0.11.0"
jupyter = ">=1.0.0"
jupytext = ">=1.14.0"
fastapi = ">=0.68.0"
uvicorn = ">=0.15.0"
pydantic = ">=1.8.0"
python-multipart = ">=0.0.5"

[dev-packages]
pytest = ">=6.0.0"
black = ">=21.0.0"
flake8 = ">=3.9.0"

[requires]
python_version = "3.9"
```

### requirements.txt (Alternative)
```txt
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
jupytext>=1.14.0
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
python-multipart>=0.0.5
```

### Installation Steps
```bash
# Using pip
pip install -r requirements.txt

# Using pipenv
pip install pipenv
pipenv install
pipenv shell

# Using conda
conda create -n movie-ml python=3.9
conda activate movie-ml
pip install -r requirements.txt
```

## 9. Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
COPY _models/ ./_models/
COPY *.csv ./data/

# Create necessary directories
RUN mkdir -p _models data

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose (docker-compose.yml)
```yaml
version: '3.8'

services:
  movie-ml-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./_models:/app/_models
      - ./data:/app/data
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped

  # Optional: Add Redis for caching
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

### Deployment Commands
```bash
# Build and run
docker build -t movie-ml-api .
docker run -p 8000:8000 movie-ml-api

# Using docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 10. FastAPI Model Serving & Testing

### FastAPI Application (main.py)
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional

app = FastAPI(title="Movie Revenue Prediction API", 
              description="API for predicting movie revenue and hit classification",
              version="1.0.0")

# Pydantic models for request validation
class MovieFeatures(BaseModel):
    runtime: float
    popularity: float
    vote_avg: float
    budget: float

class PredictionResponse(BaseModel):
    predicted_revenue: float
    predicted_revenue_millions: float
    is_hit: bool
    hit_probability: float
    budget_millions: float

# Load models
try:
    with open('_models/movie_revenue_gb_model.bin', 'rb') as f:
        reg_model = pickle.load(f)
    
    with open('classification_model.bin', 'rb') as f:
        class_model, dv = pickle.load(f)
except FileNotFoundError:
    print("Model files not found. Please ensure models are trained and saved.")
    reg_model = None
    class_model = None
    dv = None

@app.get("/")
async def root():
    return {"message": "Movie Revenue Prediction API", "status": "active"}

@app.get("/health")
async def health_check():
    models_loaded = reg_model is not None and class_model is not None
    return {
        "status": "healthy" if models_loaded else "models not loaded",
        "regression_model_loaded": reg_model is not None,
        "classification_model_loaded": class_model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_revenue(movie: MovieFeatures):
    if reg_model is None or class_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Prepare features for regression
        log_budget = np.log1p(movie.budget)
        features_df = pd.DataFrame([{
            'runtime': movie.runtime,
            'popularity': movie.popularity,
            'vote_avg': movie.vote_avg,
            'log_budget': log_budget
        }])
        
        # Regression prediction
        log_revenue_pred = reg_model.predict(features_df)[0]
        revenue_pred = np.expm1(log_revenue_pred)
        
        # Classification prediction
        feature_dict = {
            'runtime': movie.runtime,
            'popularity': movie.popularity,
            'vote_avg': movie.vote_avg,
            'log_budget': log_budget
        }
        X_class = dv.transform(feature_dict)
        hit_probability = class_model.predict_proba(X_class)[0, 1]
        is_hit = hit_probability > 0.5
        
        return PredictionResponse(
            predicted_revenue=float(revenue_pred),
            predicted_revenue_millions=float(revenue_pred / 1e6),
            is_hit=bool(is_hit),
            hit_probability=float(hit_probability),
            budget_millions=float(movie.budget / 1e6)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(movies: List[MovieFeatures]):
    predictions = []
    for movie in movies:
        prediction = await predict_revenue(movie)
        predictions.append(prediction.dict())
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### API Client Test Script (test_client.py)
```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test API health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())
    return response.status_code == 200

def test_single_prediction():
    """Test single movie prediction"""
    movie_data = {
        "runtime": 120.0,
        "popularity": 25.5,
        "vote_avg": 7.2,
        "budget": 50000000  # $50M
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=movie_data)
    
    if response.status_code == 200:
        result = response.json()
        print("Single Prediction Result:")
        print(f"  Predicted Revenue: ${result['predicted_revenue_millions']:.2f}M")
        print(f"  Budget: ${result['budget_millions']:.2f}M")
        print(f"  Is Hit: {result['is_hit']}")
        print(f"  Hit Probability: {result['hit_probability']:.2%}")
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def test_batch_predictions():
    """Test batch predictions with multiple movies"""
    movies = [
        {
            "runtime": 102.0,
            "popularity": 14.79,
            "vote_avg": 7.5,
            "budget": 8000000
        },
        {
            "runtime": 118.0,
            "popularity": 28.67,
            "vote_avg": 5.7,
            "budget": 35000000
        },
        {
            "runtime": 127.0,
            "popularity": 21.69,
            "vote_avg": 6.7,
            "budget": 60000000
        }
    ]
    
    response = requests.post(f"{BASE_URL}/batch_predict", json=movies)
    
    if response.status_code == 200:
        results = response.json()
        print("\nBatch Prediction Results:")
        for i, pred in enumerate(results['predictions']):
            print(f"Movie {i+1}:")
            print(f"  Revenue: ${pred['predicted_revenue_millions']:.2f}M")
            print(f"  Budget: ${pred['budget_millions']:.2f}M")
            print(f"  Is Hit: {pred['is_hit']} (Prob: {pred['hit_probability']:.2%})")
            print()
        return results
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

if __name__ == "__main__":
    print("Testing Movie Prediction API...")
    
    # Test health endpoint
    if test_health():
        print("✅ API is healthy")
        
        # Test single prediction
        print("\n" + "="*50)
        print("Testing Single Prediction...")
        test_single_prediction()
        
        # Test batch predictions
        print("\n" + "="*50)
        print("Testing Batch Predictions...")
        test_batch_predictions()
        
    else:
        print("❌ API health check failed")
        print("Make sure the API server is running on http://localhost:8000")
```

### Running the Tests

1. **Start the API Server**:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. **Run Client Tests**:
```bash
python test_client.py
```

3. **Access API Documentation**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Example API Calls

**Single Prediction**:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "runtime": 120.0,
       "popularity": 25.5,
       "vote_avg": 7.2,
       "budget": 50000000
     }'
```

**Batch Prediction**:
```bash
curl -X POST "http://localhost:8000/batch_predict" \
     -H "Content-Type: application/json" \
     -d '[{
       "runtime": 102.0,
       "popularity": 14.79,
       "vote_avg": 7.5,
       "budget": 8000000
     }]'
```

This comprehensive setup provides a complete machine learning pipeline from data processing to model deployment with Docker and FastAPI.
<img width="468" height="641" alt="image" src="https://github.com/user-attachments/assets/a907a0d2-090e-4b15-8856-fc6e28ccfcd0" />

