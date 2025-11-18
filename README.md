# TMDB Movie Revenue Prediction & Hit Classification System

## 1. Problem Statement

A comprehensive Machine Learning project for predicting movie revenue and classifying movie success using the TMDB (The Movie Database) dataset. This project implements both regression and classification models with deployment via FastAPI and Docker.

### Regression Problem
Predict the **revenue** of movies based on various features such as budget, popularity, runtime, vote metrics, and genre information. This helps production companies and investors estimate potential returns on investment for upcoming films.

### Classification Problem
Classify whether a movie will be **financially successful** (revenue exceeds budget by a significant margin) using movie metadata. This binary classification helps stakeholders make go/no-go decisions on film projects.

---

## 2. Installation and Running the Project

- Run the setup.sh script to set up the virtual environment and install dependencies
- Activate the virtual environment and run the training scripts


### Installation Steps
```
git clone https://github.com/clicksuku/skpmlzoomcamp_course_2025.git
```

```
bash -v setup.sh
```

### Local Deployment and Run
```bash
source mlenv/bin/activate
```

```bash
cd Script
```

```bash
uvicorn api_model_server:app --host 0.0.0.0 --port 8000
```

### Local Testing
```bash
source mlenv/bin/activate
```

```bash
cd Script
```

```bash
python api_client.py
```

### Docker Deployment and testing

### Deployment Commands
Go to the path where Dockerfile is present


```bash
cd <DOCKER FOLDER>
```


```bash
# Build
docker build -t skpmlzoomcamp .
```

```bash
# Run
docker run -p 8000:8000 skpmlzoomcamp:latest

```

### Testing

```bash
source mlenv/bin/activate
```

```bash
cd Script
```

```bash
python api_client.py
```



-----

## 3. Dataset Details and Project Files Details

### Project Files

- `setup.sh` - Script to set up the virtual environment and install dependencies
- `README.md` - Project documentation
- `requirements.txt` - List of Python packages required for the project
- `Data/` - Directory containing the dataset files
- `_models/` - Directory containing the trained models. This is created by running the setup.sh scripts
- `Notebook/` - Directory containing Jupyter notebooks for data analysis and model development
- `Script/` - Directory containing the scripts for training and prediction
- `mlenv/` - Virtual environment directory

### Data Source
- **Dataset**: TMDB 5000 Movies Dataset
- **Files**: 
  - `5kMovies.csv` - Movie metadata
  - `5kCredits.csv` - Credits information
- **Total Records**: ~4,800 movies after data cleanup

### Dataset in Columns

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

### EDA and Data Preprocessing
- **Missing Value Handling**: Categorical features with missing values removed; numerical features filled with 0
- **Genre Normalization**: JSON genre data converted to binary features for 6 major genres (Action, Adventure, Fantasy, Science Fiction, Thriller, Comedy)
- **Date Processing**: Release dates converted to release year
- **Log Transformation**: Applied to budget and revenue for better model performance
- **Feature Engineering**: Created binary success indicator for classification



## 4. Regression Model

### Target Variable
- `log_revenue` (log-transformed revenue for better model performance)

### Features Used
- `runtime`, `popularity`, `vote_avg`, `log_budget`

### Models Implemented
1. **Linear Regression** (Baseline)
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**
4. **XGBoost Regressor**

### Feature Selection
- **Numerical Features**: `log_budget`, `popularity`, `runtime`, `vote_avg`, `vote_count`, `release_year`
- **Categorical Features**: Genre binary indicators **(Not used)**

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
### Key Findings
- Log-transformed budget shows strongest correlation (0.657) with revenue
- Random Forest identifies `popularity` as most important feature (0.416)
- Feature importance differs between linear and tree-based models due to their ability to capture non-linear relationships


## 4. Model Comparison & Selection

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

### Objective
Classify movies as "successful" or "unsuccessful" based on revenue-to-budget ratio.

### Target Variable Creation
```python
# Success defined as revenue > 1.5 × budget
df['success'] = (df['revenue'] > df['budget'] * 1.5).astype(int)
```

### Target Variable
- `is_hit`: Binary classification (True/False)

### Features Used
- `runtime`, `popularity`, `vote_avg`, `log_budget`

### Models Implemented
1. **Logistic Regression** - Linear classifier with regularization
2. **Decision Tree Classifier** - Non-linear decision boundaries

### Performance Metrics
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC Score**: Area under the ROC curve
- **Confusion Matrix**: True/False Positives and Negatives
- **Classification Report**: Precision, recall, F1 for each class

### Evaluation Metrics
- **AUC-ROC**: Area Under ROC Curve
- **Precision-Recall**: Trade-off analysis
- **F1-Score**: Balanced metric for imbalanced data

## 6. K-Fold Cross Validation for Optimal C

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

### Implementation
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

---

## 7. Converting Jupyter Notebooks to Python Scripts with Jupytext
### Installation

```bash
pip install jupytext
```

### Conversion Commands

**Convert notebook to Python script (light format)**:
```bash
jupytext --to py SKP_MidTerm_Project_Regression.ipynb
jupytext --to py SKP_MidTerm_Project_Classification.ipynb
```

### Output Files
- `SKP_MidTerm_Project_Regression.py`
- `SKP_MidTerm_Project_Classification.py`

---

## 8. Requirements & Installation

### Requirements.txt
```txt
pandas
numpy
scikit-learn
scipy
threadpoolctl
jupytext
seaborn
uv
xgboost
fastapi
requests
uvicorn
tqdm
```


## 9. FastAPI Model Serving & Testing

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

## 10. Kubectl deployment local

## Kubernetes manifests (copy/paste)

Create the `k8s/` folder and add these YAMLs.

### 1 Revenue Deployment + Service (`k8s/revenue-deployment.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tmdb-revenue
  labels:
    app: tmdb-revenue
spec:
  replicas: 2
  selector:
    matchLabels:
      app: tmdb-revenue
  template:
    metadata:
      labels:
        app: tmdb-revenue
    spec:
      containers:
      - name: tmdb-revenue
        image: tmdb/revenue:v1
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
        resources:
          limits:
            cpu: "1"
            memory: "1Gi"
          requests:
            cpu: "0.25"
            memory: "256Mi"
        # mount models from a volume if you prefer
        # volumeMounts:
        # - name: models
        #   mountPath: /models
      # volumes:
      # - name: models
      #   hostPath:
      #     path: /home/you/models/revenue   # local path (minikube only)
      #     type: Directory
```

### 2 Revenue Service (`k8s/revenue-service.yaml`)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: tmdb-revenue-svc
spec:
  selector:
    app: tmdb-revenue
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: NodePort   # use ClusterIP in production, NodePort useful for local
```

### 3 Hit Deployment + Service (`k8s/hit-deployment.yaml` and `k8s/hit-service.yaml`)

Replace names and image:

```yaml
# hit-deployment.yaml (same structure as revenue but image tmdb/hit:v1, name tmdb-hit)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tmdb-hit
  labels:
    app: tmdb-hit
spec:
  replicas: 2
  selector:
    matchLabels:
      app: tmdb-hit
  template:
    metadata:
      labels:
        app: tmdb-hit
    spec:
      containers:
      - name: tmdb-hit
        image: tmdb/hit:v1
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
        resources:
          limits:
            cpu: "1"
            memory: "1Gi"
          requests:
            cpu: "0.25"
            memory: "256Mi"
```

`hit-service.yaml` same as revenue-service but name `tmdb-hit-svc` and selector `app: tmdb-hit`.



# 4. Apply manifests & run

From project root:

```bash
kubectl apply -f k8s/revenue-deployment.yaml
kubectl apply -f k8s/revenue-service.yaml
kubectl apply -f k8s/hit-deployment.yaml
kubectl apply -f k8s/hit-service.yaml

# optional ingress
kubectl apply -f k8s/ingress.yaml

# optional hpa
kubectl apply -f k8s/hpa.yaml
```

Check pods:

```bash
kubectl get pods
kubectl get svc
kubectl get deploy
kubectl describe pod <pod-name>  # debug
kubectl logs <pod-name> -c tmdb-revenue
```

---

## If using **NodePort** (services above)

Find NodePort:

```bash
kubectl get svc tmdb-revenue-svc -o yaml
# or:
kubectl get svc
```

```bash
minikube service tmdb-revenue-svc --url
# returns URL like http://192.168.xx.xx:30080
```

```bash
curl -X POST http://<ip>:<nodeport>/predict_revenue -H "Content-Type: application/json" \
  -d '{"budget":10000000,"popularity":12.3,"runtime":120,"vote_average":7.8,"vote_count":5000}'
```

```bash
curl -X POST http://tmdb.local/revenue/predict_revenue -H "Content-Type: application/json" -d '...'
```


## 11. Evaluation 


---

# 🧠 Project Evaluation Rubric

## **Problem Description**

| Points | Description                                                                                                             | Status |
| :----: | :---------------------------------------------------------------------------------------------------------------------- | :----: |
|    0   | Problem is not described                                                                                                |        |
|    1   | Problem is described in README briefly without much details                                                             |    |
|    2   | Problem is described in README with enough context, so it's clear what the problem is and how the solution will be used |   ✅   |

---

## **Exploratory Data Analysis (EDA)**

| Points | Description                                                                                                                                                                                                       | Status |
| :----: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----: |
|    0   | No EDA                                                                                                                                                                                                            |        |
|    1   | Basic EDA (looking at min–max values, checking for missing values)                                                                                                                                                |    |
|    2   | Extensive EDA (ranges of values, missing values, analysis of target variable, feature importance analysis). <br>For images: analyzing the content of the images. <br>For texts: frequent words, word clouds, etc. |  ✅   |

---

## **Model Training**

| Points | Description                                                                                                                                                           | Status |
| :----: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----: |
|    0   | No model training                                                                                                                                                     |        |
|    1   | Trained only one model, no parameter tuning                                                                                                                           |        |
|    2   | Trained multiple models (linear and tree-based). For neural networks: tried multiple variations – with dropout or without, with extra inner layers or without         |    |
|    3   | Trained multiple models and tuned their parameters. For neural networks: same as previous, but also with tuning (learning rate, dropout rate, inner layer size, etc.) |  ✅   |

---

## **Exporting Notebook to Script**

| Points | Description                                                       | Status |
| :----: | :---------------------------------------------------------------- | :----: |
|    0   | No script for training a model                                    |        |
|    1   | The logic for training the model is exported to a separate script |    ✅   |

---

## **Reproducibility**

| Points | Description                                                                                                                                                                                     | Status |
| :----: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----: |
|    0   | Not possible to execute the notebook and the training script. Data is missing or not easily accessible                                                                                          |        |
|    1   | It's possible to re-execute the notebook and the training script without errors. The dataset is committed in the project repository or there are clear instructions on how to download the data |    ✅   |

---

## **Model Deployment**

| Points | Description                                                     | Status |
| :----: | :-------------------------------------------------------------- | :----: |
|    0   | Model is not deployed                                           |        |
|    1   | Model is deployed (with Flask, BentoML, or a similar framework) |    ✅   |

---

## **Dependency and Environment Management**

| Points | Description                                                                                                                                  | Status |
| :----: | :------------------------------------------------------------------------------------------------------------------------------------------- | :----: |
|    0   | No dependency management                                                                                                                     |        |
|    1   | Provided a file with dependencies (`requirements.txt`, `Pipfile`, `bentofile.yaml`, etc.)                                                    |    |
|    2   | Provided a file with dependencies **and** used virtual environment. README explains how to install dependencies and activate the environment |  ✅    |

---

## **Containerization**

| Points | Description                                                                                  | Status |
| :----: | :------------------------------------------------------------------------------------------- | :----: |
|    0   | No containerization                                                                          |        |
|    1   | `Dockerfile` is provided or a tool that creates a Docker image is used (e.g., BentoML)       |    |
|    2   | The application is containerized **and** README describes how to build and run the container |  ✅    |

---

## **Cloud Deployment**

| Points | Description                                                                                                            | Status |
| :----: | :--------------------------------------------------------------------------------------------------------------------- | :----: |
|    0   | No deployment to the cloud                                                                                             |        |
|    1   | Documentation clearly describes (with code) how to deploy the service to cloud or Kubernetes cluster (local or remote) |   ✅    |
|    2   | Code for cloud/Kubernetes deployment is available, with URL for testing or a video/screenshot of testing it            |     |

---




