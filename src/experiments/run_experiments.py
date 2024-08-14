import rl_experiments
import rl_algorithm
import sim_env_v4
import experiment_global_vars
import compute_metrics
import plot_predicted_advs

import pickle
import numpy as np
import pandas as pd
import copy

### experiment parameters ###
'''
### Algorithm Candidates
* 'cluster_size': 'number of users within a cluster'
* 'offline_or_online': 'whether the algorithm updates online (updates posterior) or just uses the prior that was designed offline'
'''

MAX_SEED_VAL = experiment_global_vars.MAX_SEED_VAL
NUM_TRIAL_USERS = sim_env_v4.NUM_TRIAL_USERS

def get_cluster_size(pooling_type):
    if pooling_type == "full_pooling":
        return NUM_TRIAL_USERS
    elif pooling_type == "no_pooling":
        return 1
    else:
        print("ERROR: NO CLUSTER_SIZE FOUND - ", pooling_type)

def run_experiment(exp_kwargs, exp_path, job_type):
    if job_type == "simulations":
        run_simulations(exp_kwargs, exp_path)
    elif job_type == "compute_metrics":
        run_compute_metrics(exp_path)
    else:
        print("ERROR: NO JOB_TYPE FOUND - ", job_type)

# Note: hard-coded values are the exact values used in the Oralytics MRT
def run_simulations(exp_kwargs, exp_path):
    ## HANDLING RL ALGORITHM CANDIDATE ##
    cluster_size = get_cluster_size(exp_kwargs["cluster_size"])
    offline_or_online = exp_kwargs["offline_or_online"]
    current_seed = exp_kwargs["seed"]
    state_feature = exp_kwargs["state_feature"]
    print(f"State Feature: {state_feature}")
    alg_candidate = rl_algorithm.OralyticsMRTAlg(offline_or_online)

    data_pickle_template = exp_path + '/{}_data_df.p'
    update_pickle_template = exp_path + '/{}_update_df.p'

    if cluster_size == 1:
        alg_candidates = [copy.deepcopy(alg_candidate) for _ in range(NUM_TRIAL_USERS)]
        print("SEED: ", current_seed)
        np.random.seed(current_seed)
        environment_module = sim_env_v4.SimulationEnvironmentV4(state_feature)
        data_df, update_df = rl_experiments.run_experiment(alg_candidates, environment_module)
        data_df_pickle_location = data_pickle_template.format(current_seed)
        update_df_pickle_location = update_pickle_template.format(current_seed)

        print("TRIAL DONE, PICKLING NOW")
        pd.to_pickle(data_df, data_df_pickle_location)
        pd.to_pickle(update_df, update_df_pickle_location)

    elif cluster_size == NUM_TRIAL_USERS:
        print("SEED: ", current_seed)
        np.random.seed(current_seed)
        environment_module = sim_env_v4.SimulationEnvironmentV4(state_feature)
        data_df, update_df = rl_experiments.run_incremental_recruitment_exp(alg_candidate, environment_module)
        data_df_pickle_location = data_pickle_template.format(current_seed)
        update_df_pickle_location = update_pickle_template.format(current_seed)

        print("TRIAL DONE, PICKLING NOW")
        pd.to_pickle(data_df, data_df_pickle_location)
        pd.to_pickle(update_df, update_df_pickle_location)

def run_compute_metrics(exp_path):
    env_alg_mean, env_alg_lower_25 = compute_metrics.get_metric_values(exp_path, MAX_SEED_VAL)

    avg_pickle_location = exp_path + '/avg.p'
    lower_25_pickle_location = exp_path + '/low_25.p'

    with open(avg_pickle_location, 'wb') as handle:
        pickle.dump(env_alg_mean, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(lower_25_pickle_location, 'wb') as handle:
        pickle.dump(env_alg_lower_25, handle, protocol=pickle.HIGHEST_PROTOCOL)