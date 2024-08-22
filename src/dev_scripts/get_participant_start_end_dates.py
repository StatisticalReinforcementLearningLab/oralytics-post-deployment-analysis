import pandas as pd

MRT_DATA = pd.read_csv('../../data/oralytics_mrt_data.csv')
MRT_USERS = MRT_DATA['user_id'].unique()

### HELPERS ###
def get_user_data(df, user_id):
  return df[df['user_id'] == user_id]

user_start_dates = []
user_end_dates = []
for user_id in MRT_USERS:
    user_df = get_user_data(MRT_DATA, user_id)
    user_start_dates.append(user_df['user_start_day'].unique()[0])
    user_end_dates.append(user_df['user_end_day'].unique()[0])
    
# saving values to a csv
start_end_date_df = pd.DataFrame({'user_id': MRT_USERS, 'user_start_day': user_start_dates, 'user_end_day': user_end_dates})
start_end_date_df.to_csv('../../sim_env_data/v4_start_end_dates.csv')