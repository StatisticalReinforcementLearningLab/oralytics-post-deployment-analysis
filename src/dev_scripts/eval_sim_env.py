
import pandas as pd
import numpy as np
from scipy.stats import bernoulli
from scipy.stats import poisson
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

from fitting_user_models import process_mrt_data
from create_eval_metrics import plot_ci_all_features, make_metrics_comparison_table, make_error_values_table

NUM_SIMULATIONS = 500
MRT_DATA = pd.read_csv('oralytics_mrt_data.csv')
MRT_DATA = process_mrt_data(MRT_DATA)
NON_STAT_PARAMS_DF = pd.read_csv('../../sim_env_data/v4_non_stat_zip_model_params.csv')
STATE_COLS = [column for column in MRT_DATA.columns if 'state_' in column]
SIMULATED_QUALITIES_FILE = 'simulated_qualities.csv' 
SIM_QUAL_COLS = [f'sim_quality_{i+1}' for i in range(NUM_SIMULATIONS)] # Global variable

# note: since v4 only chose zip models, these are the following parameters
def get_base_params_for_user(user):
  param_dim = 7
  param_df = NON_STAT_PARAMS_DF
  user_row = np.array(param_df[param_df['User'] == user])
  bern_base = user_row[0][2:2 + param_dim]
  poisson_base = user_row[0][2 + 2 * param_dim:2 + 3 * param_dim]

  # poisson parameters, bernouilli parameters
  return bern_base, poisson_base, None

# note: since v4 only chose zip models, these are the following parameters
def get_adv_params_for_user(user):
  param_dim = 7
  param_df = NON_STAT_PARAMS_DF
  user_row = np.array(param_df[param_df['User'] == user])
  bern_adv = user_row[0][2 + param_dim: 2 + 2 * param_dim]
  poisson_adv = user_row[0][2 + 3 * param_dim:2 + 4 * param_dim]

  return bern_adv, poisson_adv


def get_user_effect_funcs():
    # negative treatment effect means users are more likely to brush
    bern_adv_func = lambda state, adv_params: min(adv_params @ state, 0)
    # positive treatment effect means users brush more seconds
    y_adv_func = lambda state, adv_params: max(adv_params @ state, 0)

    return bern_adv_func, y_adv_func

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def construct_model_and_sample(state, action, \
                                          bern_base_params, \
                                          y_base_params, \
                                          bern_adv_params, \
                                          y_adv_params, \
                                          sigma_u, \
                                          model_type, \
                                          effect_func_bern=lambda state : 0, \
                                          effect_func_y=lambda state : 0):
  bern_linear_comp = state @ bern_base_params
  if (action == 1):
    bern_linear_comp += effect_func_bern(state, bern_adv_params)
  bern_p = 1 - sigmoid(bern_linear_comp)
  # bernoulli component
  rv = bernoulli.rvs(bern_p)
  if (rv):
      y_mu = state @ y_base_params
      if (action == 1):
          y_mu += effect_func_y(state, y_adv_params)
      if model_type == "sqrt_norm":
        # normal transform component
        sample = norm.rvs(loc=y_mu, scale=sigma_u)
        sample = sample**2

        # we round to the nearest integer to produce brushing duration in seconds
        return int(sample)
      else:
        # poisson component
        l = np.exp(y_mu)
        sample = poisson.rvs(l)*rv

        return sample

  else:
    return 0

if os.path.exists(SIMULATED_QUALITIES_FILE):
    print("Using existing simulated qualities file...")
    qualities = pd.read_csv(SIMULATED_QUALITIES_FILE)
else:
  print("Running simulations...")
  for idx, row in MRT_DATA.iterrows():
      user = row['user_id']
      state = np.array(row[STATE_COLS])
      action = row['action']

      # Retrieve the parameters for the user
      bern_base_params, y_base_params, _ = get_base_params_for_user(user)
      bern_adv_params, y_adv_params = get_adv_params_for_user(user)
      effect_func_bern, effect_func_y = get_user_effect_funcs()

      # Run simulations and store the results in new columns
      for i in range(NUM_SIMULATIONS):
          quality = construct_model_and_sample(state, action,
                                              bern_base_params,
                                              y_base_params,
                                              bern_adv_params,
                                              y_adv_params,
                                              None,
                                              'zip',
                                              effect_func_bern,
                                              effect_func_y)
          sim_quality_col = f'sim_quality_{i+1}'
          MRT_DATA.at[idx, sim_quality_col] = quality

  sim_quality = MRT_DATA.filter(regex='^sim_quality|^user_id')
  MRT_DATA['mean_sim_quality'] = sim_quality.drop(columns='user_id').mean(axis=1)

  qualities = MRT_DATA[['user_id','quality']+SIM_QUAL_COLS]
  qualities.to_csv(SIMULATED_QUALITIES_FILE, index=False)

plot_ci_all_features(qualities,SIM_QUAL_COLS)

result_table_mrt_1 = make_metrics_comparison_table(qualities).to_latex(index=False)
with open('../../tables/sim_env_metrics_comparison_tbl.txt', 'w') as f:
    f.write(result_table_mrt_1)

result_table_mrt_2 = make_error_values_table(qualities,SIM_QUAL_COLS).to_latex(index=False)
with open('../../tables/sim_env_error_values_tbl.txt', 'w') as f:
    f.write(result_table_mrt_2)