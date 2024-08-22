import pandas as pd
import numpy as np
import pymc as pm
from pymc.model import Model

def process_mrt_data(df):

  df = df.query('viability == 1')
  df['decision_date'] = pd.to_datetime(df['decision_time']).dt.date

  df['user_start_day_dt'] = pd.to_datetime(df['user_start_day']).dt.date
  df['decision_date'] = pd.to_datetime(df['decision_date'])
  df['user_start_day_dt'] = pd.to_datetime(df['user_start_day'])

  df['state_day_type'] = pd.to_datetime(df['decision_time']).apply(lambda x: 1 if x.weekday() >= 5 else 0)

  df = df.drop(columns = ['state_modif'])

  df['state_day_in_study'] = (df['decision_date'] - df['user_start_day_dt']).dt.days + 1
  df['state_day_in_study'] = (df['state_day_in_study'] - 35.5) / 34.5

  desired_order = ['user_id', 'action', 'quality', 'state_tod',
                 'state_b_bar', 'state_a_bar', 'state_app_engage', 'state_day_type',
                 'state_bias', 'state_day_in_study']
  df = df[desired_order]
  return df

def get_user_data(df, user_id):
  return df[df['user_id'] == user_id]

def get_batch_data(df, user_id):
  user_df = get_user_data(df, user_id)
  states_df = user_df.filter(regex='state_*')
  outcomes = user_df['quality']
  actions = user_df['action']

  return np.array(states_df), np.array(outcomes), np.array(actions)

MRT_DATA = pd.read_csv('../../data/oralytics_mrt_data.csv')
MRT_DATA = process_mrt_data(MRT_DATA)
MRT_USERS = MRT_DATA['user_id'].unique().tolist()

## Fitting Models
### Helpers

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def build_zip_model(X, A, Y):
  model = pm.Model()
  with Model() as model:
    d = X.shape[1]
    w_b = pm.MvNormal('w_b', mu=np.zeros(d, ), cov=np.eye(d), shape=d)
    delta_b = pm.MvNormal('delta_b', mu=np.zeros(d,), cov=np.eye(d), shape=(d,))
    w_p = pm.MvNormal('w_p', mu=np.zeros(d, ), cov=np.eye(d), shape=d)
    delta_p = pm.MvNormal('delta_p', mu=np.zeros(d,), cov=np.eye(d), shape=(d,))
    bern_term = X @ w_b + A * (X @ delta_b)
    poisson_term = X @ w_p + A * (X @ delta_p)
    R = pm.ZeroInflatedPoisson("likelihood", psi=1 - sigmoid(bern_term), mu=np.exp(poisson_term), observed=Y)

  return model

def run_zip_map_for_users(users_states, users_actions, users_rewards, num_restarts):
  model_params = {}

  for user_id in users_states.keys():
    print("FOR USER: ", user_id)
    user_states = users_states[user_id]
    d = user_states.shape[1]
    user_actions = users_actions[user_id]
    user_rewards = users_rewards[user_id]
    logp_vals = np.empty(shape=(num_restarts,))
    param_vals = np.empty(shape=(num_restarts, 4 * d))
    for seed in range(num_restarts):
      model = build_zip_model(user_states, user_actions, user_rewards)
      np.random.seed(seed)
      init_params = {'w_b': np.random.randn(d), 'delta_b': np.random.randn(d), 'w_p':  np.random.randn(d), 'delta_p': np.random.randn(d)}
      with model:
        map_estimate = pm.find_MAP(start=init_params)

      w_b = map_estimate['w_b']
      delta_b = map_estimate['delta_b']
      w_p = map_estimate['w_p']
      delta_p = map_estimate['delta_p']
      logp_vals[seed] = model.compile_logp()(map_estimate)
      param_vals[seed] = np.concatenate((w_b, delta_b, w_p, delta_p), axis=None)
    model_params[user_id] = param_vals[np.argmax(logp_vals)]

  return model_params

### Execution
users_states = {}
users_rewards = {}
users_actions = {}
for user_id in MRT_USERS:
    states, rewards, actions = get_batch_data(MRT_DATA, user_id)
    users_rewards[user_id] = rewards
    users_actions[user_id] = actions
    users_states[user_id] = states

zip_model_params = run_zip_map_for_users(users_states, users_actions, users_rewards, num_restarts=5)

## Saving Parameter Values
def create_zip_df_from_params(model_columns, zip_model_params):
    rows = []
    for user in zip_model_params.keys():
        values = zip_model_params[user]
        new_row = {'User': user}
        for i in range(1, len(model_columns)):
            new_row[model_columns[i]] = values[i - 1]
        rows.append(new_row)
    df = pd.DataFrame(rows, columns=model_columns)
    return df

non_stat_zip_model_columns = ['User', 'state_tod.Base.Bern', 'state_b_bar.norm.Base.Bern', 'state_a_bar.norm.Base.Bern', 'state_app_engage.Base.Bern', 'state_day_type.Base.Bern', 'state_bias.Base.Bern', 'state_day_in_study.Base.Bern', \
                                    'state_tod.Adv.Bern', 'state_b_bar.norm.Adv.Bern', 'state_a_bar.norm.Adv.Bern', 'state_app_engage.Adv.Bern', 'state_day_type.Adv.Bern', 'state_bias.Adv.Bern', 'state_day_in_study.Adv.Bern', \
                                    'state_tod.Base.Poisson', 'state_b_bar.norm.Base.Poisson', 'state_a_bar.norm.Base.Poisson', 'state_app_engage.Base.Poisson', 'state_day_type.Base.Poisson', 'state_bias.Base.Poisson', 'state_day_in_study.Base.Poisson', \
                                    'state_tod.Adv.Poisson', 'state_b_bar.norm.Adv.Poisson', 'state_a_bar.norm.Adv.Poisson', 'state_app_engage.Adv.Poisson', 'state_day_type.Adv.Poisson', 'state_bias.Adv.Poisson', 'state_day_in_study.Adv.Poisson']

non_stat_zip_df = create_zip_df_from_params(non_stat_zip_model_columns, zip_model_params)
non_stat_zip_df.to_csv('../../sim_env_data/v4_non_stat_zip_model_params.csv')
