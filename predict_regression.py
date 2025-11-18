# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import json
import numpy as np
import seaborn as sn
import pickle

from matplotlib import pyplot as plt
from io import StringIO
# %matplotlib inline

# %%
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, f1_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# %%
###Data Cleanup and EDA####

# %%
df_movies =  pd.read_csv('_dataset/5kMovies.csv')
df_credits = pd.read_csv('_dataset/5kCredits.csv')

# %%
df_tmdb = pd.concat([df_movies, df_credits], axis=1)

# %%
new_column_order = ['id', 'title', 'original_language','genres', 'release_date','runtime', 'popularity', 'vote_count', 'vote_average', 'budget', 'revenue']
genres_list = ['Action', 'Adventure', 'Fantasy', 'Science Fiction', 'Thriller', 'Comedy']


# %%
def data_cleanup(df, new_columns):
    df_tmdb_new = df[new_column_order]
    df_tmdb_new = df_tmdb_new.loc[:, ~df_tmdb_new.columns.duplicated()]
    
    categorical= list(df_tmdb_new.dtypes[df_tmdb_new.dtypes=='object'].index)
    numerical = list(df_tmdb_new.dtypes[df_tmdb_new.dtypes!='object'].index)
    
    for c in categorical:
        df_tmdb_new = df = df_tmdb_new[df_tmdb_new[c].notna()]
        #df_tmdb_new.loc[:,c] = df_tmdb_new.loc[:,c].fillna('NA')

    for c in numerical:
        df_tmdb_new.loc[:,c] = df_tmdb_new.loc[:,c].fillna(0)
    
    df_filtered = df_tmdb_new[df_tmdb_new['genres'].str.len() > 2]

    df_filtered = df_filtered[df_filtered['release_date'].str.len() > 2]
    df_filtered['release_date_dt'] = pd.to_datetime(df_filtered['release_date'])
    df_filtered['release_year'] = df_filtered['release_date_dt'].dt.year
    df_filtered = df_filtered.drop('release_date', axis=1)
    df_filtered = df_filtered.drop('release_date_dt', axis=1)

    df_filtered.columns = ['id', 'title', 'language','genres','runtime', 'popularity', 'vote_count', 'vote_avg', 'budget', 'revenue', 'release_year']
    
    return df_filtered


# %%
def jsontocsv(jsonstr:list):
    df = pd.read_json(StringIO(jsonstr))
    csv_string = ', '.join(df['name'].apply(str))
    return csv_string


# %%
def genres_json_to_csv(row):
    return jsontocsv(row['genres'])


# %%
def normalize_genres(df_movies):
    df_movies['genres_csv'] = df_movies.apply(genres_json_to_csv, axis=1)
    for g in genres_list:
        df_movies[g] = df_movies['genres_csv'].apply(lambda x: g in x)
    df_movies = df_movies.drop('genres_csv', axis=1)
    df_movies = df_movies.drop('genres', axis=1)
    return df_movies


# %%
df_movies = data_cleanup(df_tmdb, new_column_order)
df_movies = normalize_genres(df_movies)

# %%
df_movies.columns = df_movies.columns.str.lower()
df_movies.columns = df_movies.columns.str.replace(' ','_')
df_movies.head(3)

# %%
####### Data Preparation for Regression. Run LinearRegression #######
####### Run RandomForestRegression. Tune to arrive at best parameters  #######
####### Compare Linear regression and  RandomForestRegression with best parameters. Finalize the better model as model #######

# %%
####### Data Preparation for Regression. #######

# %%
features = ['id', 'title', 'runtime', 'popularity', 'vote_avg', 'budget']
df_movies = df_movies[(df_movies['budget'] > 0) & (df_movies['revenue'] > 0)].dropna(subset=features)

df_movies['log_budget'] = np.log1p(df_movies['budget'])
df_movies['log_revenue'] = np.log1p(df_movies['revenue'])

interested_features = ['runtime', 'popularity', 'vote_avg','log_budget']
target = 'log_revenue'

# %%
X_full = df_movies[interested_features]
y_full = df_movies[target]

X_train_val, X_test, y_train_val, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
len(X_train), len(X_val), len(X_test)

# %%
####### Run LinearRegression #######

# %%
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# %%
y_pred_linear_val = lin_reg.predict(X_val)

# %%
r2s_lin_reg = r2_score(y_val, y_pred_linear_val)
rmse_lin_reg = mean_squared_error(y_val, y_pred_linear_val) 

print("R^2 Score", r2s_lin_reg)
print("RMSE", rmse_lin_reg)

np.set_printoptions(suppress=True, precision=6)
coefficients = pd.DataFrame({'feature':interested_features, 'coefficient' : lin_reg.coef_})
print(coefficients.round(6))

# %%
####### RandomForest Regression #######

# %%
rf_regression= RandomForestRegressor(n_estimators=10,random_state=42, n_jobs=-1)
rf_regression.fit(X_train, y_train)

# %%
y_pred_rf_val = rf_regression.predict(X_val)

# %%
print("R^2 Score", r2_score(y_val, y_pred_rf_val))
print("RMSE", mean_squared_error(y_val, y_pred_rf_val))

# %%
interested_features

importances = rf_regression.feature_importances_
most_important_index = np.argmax(importances)
most_important_index
most_important_feature = interested_features[most_important_index]
print(most_important_feature)

feat_imp = pd.Series(importances, index=X_full.columns).sort_values(ascending=False)
print(feat_imp)


# %%
def random_forest_varied_depth_estimator(depth, estimator):
    rf= RandomForestRegressor(n_estimators=estimator,max_depth=depth, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2s = r2_score(y_val, y_pred)
    return rmse, r2s


# %%
def find_best_depth_estimator():
    depths = [10,15,20,25, 30, 35]
    estimators = np.arange(10,210,10)

    rmse_summary = {}
    r2s_summary = {}
    results_list = []
    
    for depth in depths:
        rmses=[]
        r2ses=[]
        for estimator in estimators:
            rmse, r2s = random_forest_varied_depth_estimator(depth, estimator)
            rmses.append(rmse)
            r2ses.append(r2s)
            results_list.append({
                'depth' : depth,
                'estimator' : estimator,
                'rmse': rmse,
                'r2s' : r2s
            })
            
        rmse_summary[depth] = np.mean(rmses)
        r2s_summary[depth] = np.mean(r2ses)

    best_depth= min(rmse_summary, key=rmse_summary.get)
    
    results_df=pd.DataFrame(results_list)
    best_results_from_max_depth = results_df[results_df['depth']==best_depth]
    
    best_result_row = best_results_from_max_depth.loc[best_results_from_max_depth['rmse'].idxmin()]
    best_estimator = int(best_result_row['estimator'])
    return best_depth, best_estimator


# %%
best_max_depth, best_estimator = find_best_depth_estimator()
print("MAX Depth : ",best_max_depth)
print("Estimator : ",best_estimator)

best_rf_regressor = RandomForestRegressor(n_estimators=best_estimator,random_state=42, n_jobs=-1, max_depth=best_max_depth)
best_rf_regressor.fit(X_train, y_train)

y_pred_best_rf_val = rf_regression.predict(X_val)

# %%
print(best_max_depth, best_estimator)
r2s_rf_reg  = r2_score(y_val, y_pred_best_rf_val)
rmse_rf_reg = mean_squared_error(y_val, y_pred_best_rf_val)

print("R2 Square of Random Forest : ", r2s_rf_reg)
print("RMSE of Random Forest : ", rmse_rf_reg)

# %%
#######  Choose between the Linear and RF Model now ##########

# %%
if(rmse_rf_reg < rmse_lin_reg):
    print("Random Forest is better")
else:
    print("Linear Regression is better")

# %%
####### Now Check for GradientBoostingRegressor Regression and hypertuning parameters #######

# %%
gb_model = GradientBoostingRegressor(
    n_estimators=100,      # Number of boosting stages
    learning_rate=0.1,     # How much each tree contributes
    max_depth=3,           # Maximum depth of each tree
    min_samples_split=2,   # Minimum samples required to split
    min_samples_leaf=1,    # Minimum samples required at leaf node
    random_state=42
)

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions
y_pred_val_gb = gb_model.predict(X_val)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val_gb))
r2 = r2_score(y_val, y_pred_val_gb)
mae = mean_absolute_error(y_val, y_pred_val_gb)

print("\nGradient Boosting Results:")
print(f"RMSE: ${rmse:.2f}M")
print(f"R² Score: {r2:.4f}")
print(f"MAE: ${mae:.2f}M")


feature_importance = pd.DataFrame({
    'feature': X_full.columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n Feature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sn.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Gradient Boosting - Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')

# %%
print("Model Type:", type(gb_model))
print("Model Parameters:", gb_model.get_params())

# %%
####### Check for XGBoost and hypertuning parameters #######

# %%
import xgboost as xgb

# %%
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # For regression tasks
    n_estimators=100,              # Number of trees
    learning_rate=0.1,             # Step size shrinkage
    max_depth=3,                   # Maximum tree depth
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_compare_xgb_vals = xgb_model.predict(X_val)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_val, y_pred_compare_xgb_vals))
r2 = r2_score(y_val, y_pred_compare_xgb_vals)
mae = mean_absolute_error(y_val, y_pred_compare_xgb_vals)

print("XGBoost Results:")
print(f"RMSE: ${rmse:.2f}M")
print(f"R² Score: {r2:.4f}")
print(f"MAE: ${mae:.2f}M")

# %%
model_params = xgb_model.get_xgb_params()
print(model_params)

# %%
#####Storing All Models ######

# %%
## Storing the Linear Regression Model

with open('_models/movie_revenue_linreg_model.bin', 'wb') as f:
    pickle.dump(lin_reg, f)

## Storing the RandomForest Regression Model

with open('_models/movie_revenue_rf_model.bin', 'wb') as f:
    pickle.dump(rf_regression, f)

## Storing the Gradient Boost Regression Model
with open('_models/movie_revenue_gb_model.bin', 'wb') as f:
    pickle.dump(gb_model, f)

## Storing the XGB Regression Model
with open('_models/movie_revenue_xgb_model.bin', 'wb') as f:
    pickle.dump(xgb_model, f)

# %%
### Comparing all the models with hyper tune parameters and plotting graphs ########

# %%
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100,random_state=42, n_jobs=-1, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42,loss='squared_error'),
    'XGB Regression' : xgb.XGBRegressor(objective='reg:squarederror',learning_rate=0.1,max_depth=3,random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_compare_vals = model.predict(X_val)
    results[name] = {
        'RMSE': np.sqrt(mean_squared_error(y_val, y_pred_compare_vals)),
        'R²': r2_score(y_val, y_pred_compare_vals),
        'MAE': mean_absolute_error(y_val, y_pred_compare_vals)
    }

    plt.figure(figsize=(12, 6))

    print("Name of Model", name)
    # Plot 1: Predictions vs Actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_val, y_pred_compare_vals, alpha=0.7, s=100)
    plt.plot([y_val.min(), y_val.max()], [y_pred_compare_vals.min(), y_pred_compare_vals.max()], 'r--', lw=2)
    plt.xlabel('Actual Revenue ($M)')
    plt.ylabel('Predicted Revenue ($M)')
    plt.title('Gradient Boosting: Actual vs Predicted')
    
    # Plot 2: Residuals
    plt.subplot(1, 2, 2)
    residuals = y_val - y_pred_compare_vals
    plt.scatter(y_pred_compare_vals, residuals, alpha=0.7, s=100)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Revenue ($M)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

# Compare results
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df.round(4))

# %%
####Now lets choose the final model from above comparison #####

# %%
chosen_model = gb_model

with open('_models/regression_model.bin', 'wb') as f:
    pickle.dump(chosen_model, f)

# %%
y_pred_final = chosen_model.predict(X_val[:5])
print("runtime", "popularity", "vote_avg", "log_budget")
output = zip(X_val[:5]['runtime'], X_val[:5]['popularity'],X_val[:5]['vote_avg'],X_val[:5]['log_budget'], y_val[:5], y_pred_final)
for item in list(output):
    print("runtime", item[0])
    print("popularity",item[1])
    print("Vote_average", item[2])
    print("log_budget", item[3])
    print("log_revenue", item[4])
    print("Budget", np.expm1(item[3])/np.power(10,6))
    print("Actual Revenue", np.expm1(item[4])/np.power(10,6))
    print("Predict Revenue", np.expm1(item[5])/np.power(10,6))
    print("\n\n")

# %%
new_data = [
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

df_new_data = pd.DataFrame(new_data)

for index,row in df_new_data.iterrows():
    print("runtime", row['runtime'])
    print("popularity", row['popularity'])
    print("Vote_average", row['vote_avg'])
    print("log_budget", row['log_budget'])
    budget=np.expm1(row['log_budget'])/np.power(10,6)
    print(f"budget: ${budget:.2f}M")
    predicted_revenue = chosen_model.predict(df_new_data)[index]
    revenue=np.expm1(predicted_revenue)/np.power(10,6)
    print(f"Revenue: ${revenue:.2f}M")
    print("\n\n")


# %%
