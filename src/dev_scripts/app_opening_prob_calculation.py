# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime

COL_NAMES = ['user_id', 'user_decision_t', 'decision_time', 'action', 'quality', 'state_tod', 'state_b_bar', 'state_a_bar', 'state_app_engage', 'state_bias']
MRT_DATA = pd.read_csv('../../data/oralytics_mrt_data.csv')
MRT_USERS = MRT_DATA['user_id'].unique()
### HELPERS ###
def get_user_data(df, user_id):
  return df[df['user_id'] == user_id]

### HELPERS ###
def get_datetime(datetime_string):
    return datetime.strptime(datetime_string, '%Y-%m-%d %H:%M:%S')

"""## App Engagement For MRT Users
---
"""

# returns a dataframe with date and prior_day_opened_app values
def get_user_app_engagement_data(user_id):
  user_df = get_user_data(MRT_DATA, user_id)[['decision_time','state_app_engage']]
  user_df['date'] = user_df['decision_time'].apply(lambda datetime_string: get_datetime(datetime_string).date())
  user_df = user_df.groupby('date', as_index=False).sum(numeric_only=True)
  user_df['prior_day_opened_app'] = user_df['state_app_engage'].apply(lambda x: 1 if x > 0 else 0)

  return user_df[["date", "prior_day_opened_app"]]

for user_id in MRT_USERS:
  app_opening_data = get_user_app_engagement_data(user_id)['prior_day_opened_app']
  open_app_idxs = np.nonzero(np.array(app_opening_data))[0]
  num_days = len(open_app_idxs)
  total_days = len(app_opening_data)
  print("{} opened their app {}/{} days" .format(user_id, num_days, total_days))

prop_open_app = []
for user_id in MRT_USERS:
  app_opening_data = get_user_app_engagement_data(user_id)['prior_day_opened_app']
  open_app_idxs = np.nonzero(np.array(app_opening_data))[0]
  open_app_prob = len(open_app_idxs) / len(app_opening_data)
  prop_open_app.append(open_app_prob)

# saving values to a csv
app_open_df = pd.DataFrame({'user_id': MRT_USERS, 'app_open_prob': prop_open_app})
app_open_df.to_csv('../../sim_env_data/v4_app_open_prob.csv')