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
Full Cross Validation Results
   param_C  mean_test_score  std_test_score  rank_test_score
0    0.001         0.780968        0.018832                6
1    0.010         0.784742        0.018426                5
2    0.100         0.799354        0.016126                4
3    1.000         0.805223        0.015208                1
4   10.000         0.804724        0.015922                2
5  100.000         0.804656        0.015973                3
```

### Optimal Parameter
- **Best C**: 1 (provides optimal regularization strength)
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
<img width="397" height="600" alt="image" src="https://github.com/user-attachments/assets/0b5da70a-7c7f-42d3-8264-c095c89b341d" />

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

### FastAPI Application (api_model_server.py)
```python
import pickle
import pandas as pd
from fastapi import FastAPI
from FilmFeature import FilmFeature

app = FastAPI()

with open('/Users/sundar/Documents/Professional/Code/gitclicksuku/skpmlzoomcamp/MidYearProject/_Models/classification_model.bin', 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
    model_classification, dv_classification = pickle.load(f_in)

with open('/Users/sundar/Documents/Professional/Code/gitclicksuku/skpmlzoomcamp/MidYearProject/_Models/regression_model.bin', 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
    model_regression = pickle.load(f_in)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict_revenue")
async def predict_revenue(request: FilmFeature):
    x = pd.DataFrame([request.dict()])
    y_pred = model_regression.predict(x)
    return {"revenue": float(y_pred)}


@app.post("/predict_hit")
async def predict_hit(request: FilmFeature):
    x = dv_classification.transform([request.dict()])
    y_pred = model_classification.predict_proba(x)[0, 1]
    #y_pred_status = model_classification.predict(x)
    return {"probability": float(y_pred)}
```

### API Client Test Script (test_client.py)
```python
import requests
import pandas as pd
import numpy as np

url_revenue = "http://127.0.0.1:8000/predict_revenue"
url_hit = "http://127.0.0.1:8000/predict_hit"


#interested_features = ['runtime', 'popularity', 'vote_avg','log_budget']

films = [
  {
    "runtime": 102.0,
    "popularity": 14.793177,
    "vote_avg": 7.5,
    "log_budget": 15.894952
  },
  {
    "runtime": 118.0,
    "popularity": 28.670477,
    "vote_avg": 5.7,
    "log_budget": 17.370859
  },
  {
    "runtime": 112.0,
    "popularity": 9.969624,
    "vote_avg": 6.3,
    "log_budget": 16.38046
  },
  {
    "runtime": 110.0,
    "popularity": 46.078371,
    "vote_avg": 5.9,
    "log_budget": 17.453097
  },
  {
    "runtime": 127.0,
    "popularity": 21.685719,
    "vote_avg": 6.7,
    "log_budget": 17.909855
  }
]

films_hit = [
  {
    "runtime": 102.0,
    "popularity": 14.793177,
    "vote_avg": 7.5,
    "log_budget": 15.894952
  },
  {
    "runtime": 118.0,
    "popularity": 28.670477,
    "vote_avg": 5.7,
    "log_budget": 17.370859
  },
  {
    "runtime": 112.0,
    "popularity": 9.969624,
    "vote_avg": 6.3,
    "log_budget": 16.38046
  },
  {
    "runtime": 110.0,
    "popularity": 46.078371,
    "vote_avg": 5.9,
    "log_budget": 17.453097
  },
  {
    "runtime": 127.0,
    "popularity": 21.685719,
    "vote_avg": 6.7,
    "log_budget": 17.909855
  }
]


df_new_data = pd.DataFrame(films)

for index,row in df_new_data.iterrows():
    print("runtime", row['runtime'])
    print("popularity", row['popularity'])
    print("Vote_average", row['vote_avg'])
    print("log_budget", row['log_budget'])
    budget=np.expm1(row['log_budget'])/np.power(10,6)
    print(f"budget: ${budget:.2f}M")
    predicted_revenue = requests.post(url_revenue, json=films[index])
    revenue=np.expm1(predicted_revenue.json()['revenue'])/np.power(10,6)
    print(f"Revenue: ${revenue:.2f}M")

    is_hit = requests.post(url_hit, json=films[index])
    probability = is_hit.json()['probability']
    print("Probability", probability)
    if probability >= 0.5:
        print("Hit")
    else:
        print("Not a Hit")
    
    print("\n\n")
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
