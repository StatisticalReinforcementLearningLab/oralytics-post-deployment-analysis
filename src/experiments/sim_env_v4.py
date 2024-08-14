# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.stats import bernoulli

import read_write_info
import simulation_environment
import reward_definition

"""## BASE ENVIRONMENT COMPONENT
---
"""

NON_STAT_PARAMS_DF = pd.read_csv(read_write_info.READ_PATH_PREFIX + 'sim_env_data/v4_non_stat_zip_model_params.csv')
APP_OPEN_PROB_DF = pd.read_csv(read_write_info.READ_PATH_PREFIX + 'sim_env_data/v4_app_open_prob.csv')
SIM_ENV_USERS = np.array(NON_STAT_PARAMS_DF['User'])
# getting participant start and end dates
START_END_DATES_DF = pd.read_csv(read_write_info.READ_PATH_PREFIX + 'sim_env_data/v4_start_end_dates.csv')
# values used by run_experiments
NUM_TRIAL_USERS = len(SIM_ENV_USERS)
TRIAL_START_DATE = "2023-09-22"
TRIAL_END_DATE = "2024-07-16"

"""### Generate States
---
"""

def get_previous_day_qualities_and_actions(j, Qs, As):
    if j > 1:
        if j % 2 == 0:
            return Qs, As
        else:
            # current evening dt does not use most recent quality or action
            return Qs[:-1], As[:-1]
    # first day return empty Qs and As back
    else:
        return Qs, As

# Stationary State Space
# 0 - time of day
# 1 - b_bar (normalized)
# 2 - a_bar (normalized)
# 3 - app engagement
# 4 - weekday vs. weekend
# 5 - bias

# Non-stationary state space
# 0 - time of day
# 1 - b_bar (normalized)
# 2 - a_bar (normalized)
# 3 - app engagement
# 4 - weekday vs. weekend
# 5 - bias
# 6 - day in study

# Note: We assume that all participants start the study on Monday (j = 0 denotes)
# Monday morning. Therefore the first weekend idx is j = 10 (Saturday morning)
def generate_env_state(j, user_qualities, user_actions, app_engagement):
    env_state = np.ones(7)
    # session type - either 0 or 1
    session_type = j % 2
    env_state[0] = session_type
    # b_bar, a_bar (normalized)
    Qs, As = get_previous_day_qualities_and_actions(j, user_qualities, user_actions)
    b_bar, a_bar = reward_definition.get_b_bar_a_bar(Qs, As)
    env_state[1] = reward_definition.normalize_b_bar(b_bar)
    env_state[2] = reward_definition.normalize_a_bar(a_bar)
    # app engagement
    env_state[3] = app_engagement
    # weekday vs. weekend
    env_state[4] = 1 if (j % 14 >= 10 and j % 14 <= 13) else 0
    # bias
    env_state[5] = 1
    # day in study if a non-stationary environment
    env_state[6] = simulation_environment.normalize_day_in_study(1 + (j // 2))

    return env_state

def get_app_open_prob(user_id):
    return APP_OPEN_PROB_DF[APP_OPEN_PROB_DF['user_id'] == user_id]['app_open_prob'].values[0]

def get_user_start_date(user_id):
    return START_END_DATES_DF[START_END_DATES_DF['user_id'] == user_id]['user_start_day'].values[0]

def get_user_end_date(user_id):
    return START_END_DATES_DF[START_END_DATES_DF['user_id'] == user_id]['user_end_day'].values[0]

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

"""## Creating Simulation Environment Objects
---
"""

class UserEnvironmentV4(simulation_environment.UserEnvironmentAppEngagement):
    def __init__(self, user_id, model_type, adv_params, \
                user_params, user_effect_func_bern, user_effect_func_y):
        # Note: in the base UserEnvironment, it uses simulated user_effect_sizes,
        # but we replace it with adv_params, user's fitted advantage parameters
        super(UserEnvironmentV4, self).__init__(user_id, model_type, adv_params, \
                            user_params, user_effect_func_bern, user_effect_func_y)
        # probability of opening app
        self.app_open_base_prob = get_app_open_prob(user_id)
        # for incremental recruitment, we use the actual start date from the Oralytics MRT
        self.start_date = get_user_start_date(user_id)
        self.end_date = get_user_end_date(user_id)

    def get_start_date(self):
        return self.start_date
    
    def get_end_date(self):
        return self.end_date

# def create_user_envs(users_list):
#     all_user_envs = {}
#     for i, user_id in enumerate(users_list):
#       model_type = "zip" # note: all users in V4 have the zero-inflated poisson model
#       base_params = get_base_params_for_user(user_id)
#       adv_params = get_adv_params_for_user(user_id)
#       user_effect_func_bern, user_effect_func_y = get_user_effect_funcs()
#       new_user = UserEnvironmentV4(user_id, model_type, adv_params, \
#                         base_params, user_effect_func_bern, user_effect_func_y)
#       all_user_envs[i] = new_user

#     return all_user_envs

# modifies the advantage parameters depending on the state_feature specified
# if state_feature is None or not one of the algorithm features then we make
# no modifications and the advantage parameters are the one fitted using the MRT data
def create_user_envs_under_no_treatment(users_list, state_feature):
    all_user_envs = {}
    for i, user_id in enumerate(users_list):
      model_type = "zip" # note: all users in V4 have the zero-inflated poisson model
      base_params = get_base_params_for_user(user_id)
      adv_params = get_adv_params_for_user(user_id)
      # modify adv_params based on the state_feature
      if state_feature == "tod":
          adv_params[0][0] = 0
          adv_params[1][0] = 0
      elif state_feature == "b_bar":
          adv_params[0][1] = 0
          adv_params[1][1] = 0
      elif state_feature == "a_bar":
          adv_params[0][2] == 0
          adv_params[1][2] == 0
      elif state_feature == "app_engage":
          adv_params[0][3] == 0
          adv_params[1][3] == 0
      # for the bias, we zero out all advantage parameters
      elif state_feature == "bias":
          adv_params = np.zeros(7), np.zeros(7)
      user_effect_func_bern, user_effect_func_y = get_user_effect_funcs()
      new_user = UserEnvironmentV4(user_id, model_type, adv_params, \
                        base_params, user_effect_func_bern, user_effect_func_y)
      all_user_envs[i] = new_user

    return all_user_envs

# Environment used to re-evaluate algorithm design decisions
class SimulationEnvironmentV4(simulation_environment.SimulationEnvironmentAppEngagement):
    def __init__(self, state_feature=None):
        # note: v4 uses the exact 79 participants from the Oralytics MRT
        user_envs = create_user_envs_under_no_treatment(SIM_ENV_USERS, state_feature)

        super(SimulationEnvironmentV4, self).__init__(SIM_ENV_USERS, user_envs)

        self.version = "V4"
        self.trial_start_date = TRIAL_START_DATE
        self.trial_end_date = TRIAL_END_DATE

    def get_trial_start_end_dates(self):
        return self.trial_start_date, self.trial_end_date

    def generate_current_state(self, user_idx, j):
        # prior day app_engagement is 0 for the first day
        prior_app_engagement = self.get_user_prior_day_app_engagement(user_idx)
        self.simulate_app_opening_behavior(user_idx, j)
        brushing_qualities = np.array(self.get_env_history(user_idx, "outcomes"))
        past_actions = np.array(self.get_env_history(user_idx, "actions"))

        return generate_env_state(j, brushing_qualities, past_actions, prior_app_engagement)
