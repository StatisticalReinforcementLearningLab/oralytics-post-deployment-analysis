import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd

import read_write_info
import experiment_global_vars

READ_PATH_PREFIX = read_write_info.READ_PATH_PREFIX
WRITE_PATH_PREFIX = read_write_info.WRITE_PATH_PREFIX + "figures/"
MAX_SEED_VAL = experiment_global_vars.MAX_SEED_VAL
NUM_ALG_UPDATES = experiment_global_vars.NUM_ALG_UPDATES
POSTERIOR_DF = pd.read_csv(READ_PATH_PREFIX + '/sim_env_data/posterior_weights_table.csv', index_col=0)
POSTERIOR_DF = POSTERIOR_DF.drop(columns=['timestamp'])

## HELPER FUNCTION ##
### GETTING POSTERIOR ###
D_ADVANTAGE = 5
RL_ALG_FEATURE_DIM = 15
def get_posterior_values(df, policy_idx):
  row = df[df['policy_idx'] == policy_idx]
  row_vals = np.array(row.iloc[:, 1:].values).reshape(-1, 1)
  beta_mean = row_vals[:RL_ALG_FEATURE_DIM][-D_ADVANTAGE:].flatten()
  beta_var = row_vals[RL_ALG_FEATURE_DIM:].reshape(RL_ALG_FEATURE_DIM, RL_ALG_FEATURE_DIM)[-D_ADVANTAGE:,-D_ADVANTAGE:]

  return beta_mean, beta_var

### COMPUTING POSTERIOR STATISTIC ###
def compute_posterior_stat(beta_mean, beta_var, c):
  return (beta_mean @ c) / np.sqrt(c @ beta_var @ c)

def calculate_predicted_advs(posterior_df, policy_idxs, state):
    result = []
    for policy_idx in policy_idxs:
        beta_mean, beta_var = get_posterior_values(posterior_df, policy_idx)
        result.append(compute_posterior_stat(beta_mean, beta_var, state))

    return result

def plot_stat_across_time(x, trial_trajectory, resampled_trajectories, title, save_as_pdf=False, pdf_filename="test"):
    plt.figure(figsize=(10, 6))
    plt.violinplot(resampled_trajectories, showmeans=False, showmedians=True)
    plt.plot(x, trial_trajectory, marker='o', linestyle='-', color='b')
    plt.title(title, fontsize=25)
    plt.xlabel("Update Times", fontsize=20)
    plt.xticks(fontsize=15)
    # need to re-index because plt.violinplot defaults indexing starting at 1
    plt.xticks(ticks=range(1, len(x) + 1, 5), labels=range(0, len(x), 5))
    plt.ylabel('Predicted Advantage (Standardized)', fontsize=20)
    plt.yticks(fontsize=15)
    if save_as_pdf:
        plt.savefig(pdf_filename + ".pdf", format='pdf')
        print(f"Plot saved as {pdf_filename}")
    else:
        plt.show()

## GETTING PREDICTIVE ADVANTAGE FOR EACH POSTERIOR VALUE FROM RE-SAMPLING METHOD ##
def get_actual_trial_pred_advs_for_state(state):
    policy_idxs = list(POSTERIOR_DF['policy_idx'])
    pred_adv_across_time = calculate_predicted_advs(POSTERIOR_DF, policy_idxs, np.array(state))

    return pred_adv_across_time

def get_pred_advs_for_state(string_prefix, state):
  pred_advs = np.zeros(shape=(MAX_SEED_VAL, NUM_ALG_UPDATES))
  for seed in range(MAX_SEED_VAL):
    try:
        pickle_name = string_prefix + f"{seed}_update_df.p"
        update_df = pd.read_pickle(pickle_name)
        update_df.rename(columns={'update_t': 'policy_idx'}, inplace=True)
        policy_idxs = list(update_df['policy_idx'])
        pred_adv_across_time = calculate_predicted_advs(update_df, policy_idxs, np.array(state))
        pred_advs[seed] = pred_adv_across_time
    except Exception as e:
        print(f"Couldn't for {pickle_name}")
        print(f"Error: {e}")

  return pred_advs

## STATES ##
# # Define the possible values for each dimension
dimension1 = [0, 1] # time of day
dimension2 = [-0.7, 0.1] # b_bar
dimension3 = [-0.6, -0.1] # a_bar
dimension4 = [0, 1] # prior day app engagement
dimensions = [dimension1, dimension2, dimension3, dimension4]
combinations = list(itertools.product(*dimensions))

for state in combinations:
  string_prefix = READ_PATH_PREFIX + f"did_we_learn/full_pooling_online_{list(state)}/"
  print(f"getting results from {string_prefix}")
  alg_state = list(state)
  alg_state.append(1) # adding intercept back in
  pred_adv_across_time = get_actual_trial_pred_advs_for_state(alg_state)
  resampled_trajectories = get_pred_advs_for_state(string_prefix, alg_state)
  # note: x-axis has to be shifted by 1 because violin plots start indexing by 1 even though we coded it to start by 0
  plot_stat_across_time(range(1, NUM_ALG_UPDATES + 1), pred_adv_across_time, resampled_trajectories, f"f(s) = {alg_state}", True, WRITE_PATH_PREFIX + "did_we_learn/" + str(alg_state))