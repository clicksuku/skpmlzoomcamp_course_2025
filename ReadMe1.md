# TMDB Movie Revenue Prediction & Classification

A comprehensive Machine Learning project for predicting movie revenue and classifying movie success using the TMDB (The Movie Database) dataset. This project implements both regression and classification models with deployment via FastAPI and Docker.

---

## 1. Problem Statement

### Regression Problem
Predict the **revenue** of movies based on various features such as budget, popularity, runtime, vote metrics, and genre information. This helps production companies and investors estimate potential returns on investment for upcoming films.

### Classification Problem
Classify whether a movie will be **financially successful** (revenue exceeds budget by a significant margin) using movie metadata. This binary classification helps stakeholders make go/no-go decisions on film projects.

---

## 2. Dataset Details

### Data Source
- **Dataset**: TMDB 5000 Movies Dataset
- **Files**: 
  - `5kMovies.csv` - Movie metadata
  - `5kCredits.csv` - Credits information
- **Total Records**: ~4,800 movies after data cleanup

### Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `id` | Integer | Unique movie identifier |
| `title` | String | Movie title |
| `original_language` | String | Language of the movie |
| `genres` | JSON/String | Movie genres (normalized to binary features) |
| `release_date` | Date | Release date (converted to release_year) |
| `runtime` | Float | Movie duration in minutes |
| `popularity` | Float | Popularity score |
| `vote_count` | Integer | Number of votes |
| `vote_average` | Float | Average rating |
| `budget` | Integer | Production budget |
| `revenue` | Integer | Total revenue (target for regression) |

### Data Preprocessing
- **Missing Value Handling**: Categorical features with missing values removed; numerical features filled with 0
- **Genre Normalization**: JSON genre data converted to binary features for 6 major genres (Action, Adventure, Fantasy, Science Fiction, Thriller, Comedy)
- **Date Processing**: Release dates converted to release year
- **Log Transformation**: Applied to budget and revenue for better model performance
- **Feature Engineering**: Created binary success indicator for classification

---

## 3. Regression Model

### Objective
Predict movie revenue using numerical and categorical features.

### Models Implemented
1. **Linear Regression** - Baseline model
2. **Random Forest Regressor** - Ensemble method
3. **Gradient Boosting Regressor** - Advanced ensemble method

### Feature Selection
- **Numerical Features**: `log_budget`, `popularity`, `runtime`, `vote_avg`, `vote_count`, `release_year`
- **Categorical Features**: Genre binary indicators

### Performance Metrics
- **R² Score**: Measures variance explained by the model
- **RMSE (Root Mean Squared Error)**: Average prediction error
- **MAE (Mean Absolute Error)**: Average absolute difference

### Key Findings
- Log-transformed budget shows strongest correlation (0.657) with revenue
- Random Forest identifies `popularity` as most important feature (0.416)
- Feature importance differs between linear and tree-based models due to their ability to capture non-linear relationships

---

## 4. Model Selection Process

### Comparison Framework

The optimal regression model was selected through systematic comparison:

1. **Linear Regression** (Baseline)
   - Simple interpretability
   - Assumes linear relationships
   - Best for understanding feature correlations

2. **Random Forest Regressor**
   - Handles non-linear relationships
   - Provides feature importance scores
   - Reduces overfitting through ensemble averaging
   - **Parameters tuned**: `n_estimators`, `max_depth`, `min_samples_split`

3. **Gradient Boosting Regressor**
   - Sequential error correction
   - Often achieves best performance
   - **Parameters tuned**: `learning_rate`, `n_estimators`, `max_depth`

### Hyperparameter Tuning with GridSearchCV

```python
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
```

### Model Selection Criteria
- **Cross-validation score** (5-fold CV)
- **Test set performance** (RMSE, R², MAE)
- **Training time** and **prediction speed**
- **Feature importance interpretability**

---

## 5. Classification Model

### Objective
Classify movies as "successful" or "unsuccessful" based on revenue-to-budget ratio.

### Target Variable Creation
```python
# Success defined as revenue > 1.5 × budget
df['success'] = (df['revenue'] > df['budget'] * 1.5).astype(int)
```

### Models Implemented
1. **Logistic Regression** - Linear classifier with regularization
2. **Decision Tree Classifier** - Non-linear decision boundaries

### Performance Metrics
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC Score**: Area under the ROC curve
- **Confusion Matrix**: True/False Positives and Negatives
- **Classification Report**: Precision, recall, F1 for each class

---

## 6. KFold Cross-Validation for Optimal Regularization Parameter (C)

### Why KFold?
K-Fold cross-validation ensures robust model evaluation by:
- Preventing overfitting to a single train-test split
- Utilizing the entire dataset for both training and validation
- Providing more stable performance estimates

### Implementation for Logistic Regression

```python
from sklearn.model_selection import KFold

# Define K-Fold with 5 splits
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Test different C values (inverse of regularization strength)
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

for C in C_values:
    scores = []
    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_train, y_train)
        score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        scores.append(score)
    
    print(f"C={C}: Mean AUC = {np.mean(scores):.4f}")
```

### Optimal C Selection
- **Smaller C**: Stronger regularization (simpler model, less overfitting)
- **Larger C**: Weaker regularization (more complex model, potential overfitting)
- **Optimal C**: Value that maximizes mean validation AUC across all folds

### Result Interpretation
The C value with the highest mean cross-validation score is selected for the final model, ensuring generalization to unseen data.

---

## 7. Converting Jupyter Notebooks to Python Scripts with Jupytext

### Installation

```bash
pip install jupytext
```

### Conversion Commands

**Convert notebook to Python script (percent format)**:
```bash
jupytext --to py:percent SKP_MidTerm_Project_Regression.ipynb
jupytext --to py:percent SKP_MidTerm_Project_Classification.ipynb
```

**Convert notebook to Python script (light format)**:
```bash
jupytext --to py SKP_MidTerm_Project_Regression.ipynb
jupytext --to py SKP_MidTerm_Project_Classification.ipynb
```

**Alternative using nbconvert**:
```bash
jupyter nbconvert --to script SKP_MidTerm_Project_Regression.ipynb
jupyter nbconvert --to script SKP_MidTerm_Project_Classification.ipynb
```

### Output Files
- `SKP_MidTerm_Project_Regression.py`
- `SKP_MidTerm_Project_Classification.py`

### Advantages of Jupytext
- Preserves cell structure with `# %%` markers
- Maintains markdown comments as docstrings
- Allows bidirectional conversion (py ↔ ipynb)
- Better version control compatibility

---

## 8. Requirements & Virtual Environment Setup

### Option 1: Using Pipenv (Recommended)

**Install Pipenv**:
```bash
pip install pipenv
```

**Create virtual environment and install dependencies**:
```bash
# Navigate to project directory
cd tmdb-ml-project

# Create Pipfile and install packages
pipenv install pandas numpy scikit-learn matplotlib seaborn fastapi uvicorn pydantic python-multipart

# For development dependencies
pipenv install --dev jupytext jupyter tqdm

# Activate virtual environment
pipenv shell
```

**Pipfile Example**:
```toml
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
pandas = "*"
numpy = "*"
scikit-learn = "*"
matplotlib = "*"
seaborn = "*"
fastapi = "*"
uvicorn = "*"
pydantic = "*"
python-multipart = "*"

[dev-packages]
jupytext = "*"
jupyter = "*"
tqdm = "*"

[requires]
python_version = "3.10"
```

**Install from Pipfile**:
```bash
pipenv install
```

**Export to requirements.txt**:
```bash
pipenv requirements > requirements.txt
```

### Option 2: Using pip and requirements.txt

**Create virtual environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Install dependencies**:
```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
fastapi==0.103.1
uvicorn==0.23.2
pydantic==2.3.0
python-multipart==0.0.6
jupytext==1.15.0
tqdm==4.66.1
```

---

## 9. Docker Structure & Deployment

### Project Structure

```
tmdb-ml-project/
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── models/
│   │   ├── regression_model.pkl
│   │   └── classification_model.pkl
│   └── utils.py                # Helper functions
│
├── data/
│   ├── 5kMovies.csv
│   └── 5kCredits.csv
│
├── notebooks/
│   ├── SKP_MidTerm_Project_Regression.ipynb
│   └── SKP_MidTerm_Project_Classification.ipynb
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Pipfile
├── Pipfile.lock
└── README.md
```

### Dockerfile

```dockerfile
# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app ./app
COPY ./data ./data

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Building and Running Docker Container

**Build the Docker image**:
```bash
docker build -t tmdb-ml-api .
```

**Run the container**:
```bash
docker run -d -p 8000:8000 --name tmdb-api tmdb-ml-api
```

**View logs**:
```bash
docker logs tmdb-api
```

**Stop container**:
```bash
docker stop tmdb-api
```

**Remove container**:
```bash
docker rm tmdb-api
```

### Docker Compose (Optional)

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  api:
    build: .
    container_name: tmdb-ml-api
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app/app
      - ./data:/app/data
    environment:
      - ENV=production
    restart: unless-stopped
```

**Run with Docker Compose**:
```bash
docker-compose up -d
```

---

## 10. FastAPI Model Serving & Testing

### FastAPI Application Structure

**app/main.py**:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI(title="TMDB Movie Prediction API", version="1.0")

# Load models
with open('app/models/regression_model.pkl', 'rb') as f:
    regression_model = pickle.load(f)

with open('app/models/classification_model.pkl', 'rb') as f:
    classification_model = pickle.load(f)

# Request schema
class MovieInput(BaseModel):
    budget: float
    popularity: float
    runtime: float
    vote_avg: float
    vote_count: int
    release_year: int
    action: bool = False
    adventure: bool = False
    fantasy: bool = False
    science_fiction: bool = False
    thriller: bool = False
    comedy: bool = False

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "TMDB ML API is running", "status": "healthy"}

# Regression prediction endpoint
@app.post("/predict/revenue")
def predict_revenue(movie: MovieInput):
    # Prepare features
    features = pd.DataFrame([movie.dict()])
    features['log_budget'] = np.log1p(features['budget'])
    
    # Make prediction
    log_revenue_pred = regression_model.predict(features)
    revenue_pred = np.expm1(log_revenue_pred[0])
    
    return {
        "predicted_revenue": float(revenue_pred),
        "log_revenue": float(log_revenue_pred[0])
    }

# Classification prediction endpoint
@app.post("/predict/success")
def predict_success(movie: MovieInput):
    # Prepare features
    features = pd.DataFrame([movie.dict()])
    
    # Make prediction
    success_pred = classification_model.predict(features)[0]
    success_proba = classification_model.predict_proba(features)[0]
    
    return {
        "success": bool(success_pred),
        "probability_failure": float(success_proba[0]),
        "probability_success": float(success_proba[1])
    }
```

### Running the API Locally

```bash
# Activate virtual environment
pipenv shell  # or: source .venv/bin/activate

# Run FastAPI with Uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Testing the API

**1. Health Check**:
```bash
curl http://localhost:8000/
```

**2. Predict Revenue**:
```bash
curl -X POST "http://localhost:8000/predict/revenue" \
  -H "Content-Type: application/json" \
  -d '{
    "budget": 100000000,
    "popularity": 25.5,
    "runtime": 120,
    "vote_avg": 7.2,
    "vote_count": 5000,
    "release_year": 2024,
    "action": true,
    "adventure": true
  }'
```

**3. Predict Success**:
```bash
curl -X POST "http://localhost:8000/predict/success" \
  -H "Content-Type: application/json" \
  -d '{
    "budget": 50000000,
    "popularity": 18.3,
    "runtime": 110,
    "vote_avg": 6.8,
    "vote_count": 3000,
    "release_year": 2024,
    "comedy": true
  }'
```

### Python Client Example

```python
import requests

url = "http://localhost:8000/predict/revenue"
movie_data = {
    "budget": 150000000,
    "popularity": 35.7,
    "runtime": 135,
    "vote_avg": 7.8,
    "vote_count": 8000,
    "release_year": 2024,
    "action": True,
    "science_fiction": True
}

response = requests.post(url, json=movie_data)
print(response.json())
```

### Docker API Testing

```bash
# Test API running in Docker container
curl http://localhost:8000/

curl -X POST "http://localhost:8000/predict/revenue" \
  -H "Content-Type: application/json" \
  -d '{"budget": 100000000, "popularity": 25.5, "runtime": 120, "vote_avg": 7.2, "vote_count": 5000, "release_year": 2024, "action": true}'
```

---

## Project Workflow Summary

1. **Data Preparation**: Load and clean TMDB datasets
2. **EDA**: Explore correlations, distributions, and feature relationships
3. **Feature Engineering**: Log transformations, genre encoding, success labels
4. **Model Training**: 
   - Regression: Linear, Random Forest, Gradient Boosting
   - Classification: Logistic Regression with KFold CV, Decision Tree
5. **Model Evaluation**: Compare metrics and select best models
6. **Export Models**: Save trained models as `.pkl` files
7. **Notebook to Script**: Convert `.ipynb` to `.py` using Jupytext
8. **API Development**: Build FastAPI endpoints for predictions
9. **Dockerization**: Create container for deployment
10. **Testing**: Validate API endpoints locally and in Docker

---

## Key Learnings

- **Feature Importance Varies by Model**: Linear models prioritize linearly correlated features (budget), while tree-based models capture complex interactions (popularity)
- **Cross-Validation is Critical**: KFold CV provides robust hyperparameter selection
- **Log Transformations**: Essential for skewed distributions (revenue, budget)
- **Deployment Ready**: FastAPI + Docker enables production-grade ML serving
- **Reproducibility**: Virtual environments and containerization ensure consistent results

---

## Future Enhancements

- Add XGBoost for potentially better performance
- Implement ensemble methods combining multiple models
- Add feature selection techniques (RFE, LASSO)
- Deploy to cloud platforms (AWS Lambda, Azure, GCP)
- Add monitoring and logging for production
- Implement A/B testing framework for model updates
- Create web interface for non-technical users

---

## Contributing

Contributions are welcome! Please submit issues or pull requests for improvements.

## License

MIT License

---

**Author**: Machine Learning Student  
**Project Type**: Capstone Project  
**Date**: November 2025
