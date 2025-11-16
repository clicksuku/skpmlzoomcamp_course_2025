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