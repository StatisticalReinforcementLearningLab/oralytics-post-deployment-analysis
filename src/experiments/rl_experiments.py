import numpy as np
import pandas as pd
import reward_definition
import experiment_global_vars
from datetime import datetime, timedelta


TRIAL_LENGTH_IN_WEEKS = experiment_global_vars.TRIAL_LENGTH_IN_WEEKS
NUM_DECISION_TIMES = experiment_global_vars.NUM_DECISION_TIMES
FILL_IN_COLS = experiment_global_vars.FILL_IN_COLS

# helpers for recruitment by date
def get_date(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d').date()

def increment_date(date_string):
    next_date = get_date(date_string) + timedelta(days=1)
    return next_date.strftime("%Y-%m-%d") 

def dt_to_user_day_in_study(dt):
    return dt // 2 + 1

def calculate_day_in_study(start_date_str, curr_date_str):
    curr_date =  get_date(curr_date_str)
    start_date =  get_date(start_date_str)

    return (curr_date - start_date).days + 1

def get_morning_decision_t(user_day_in_study):
    return (2 * user_day_in_study) - 2

def get_evening_decision_t(user_day_in_study):
    return (2 * user_day_in_study) - 1

# assumes a weekly recruitment rate
def compute_num_updates(users_groups, update_cadence):
    last_group_idx = max(users_groups[:,1].astype(int))
    num_study_decision_times = (last_group_idx + TRIAL_LENGTH_IN_WEEKS) * 7 * 2
    # we subtract 1 because we do not update after the final week of the study
    num_updates = num_study_decision_times / update_cadence

    return int(num_updates)

def create_dfs_no_pooling(users, update_cadence, rl_algorithm_feature_dim):
    N = len(users)
    batch_data_size = N * NUM_DECISION_TIMES
    ### data df ###
    data_dict = {}
    data_dict['user_idx'] = np.repeat(range(N), NUM_DECISION_TIMES)
    data_dict['user_id'] = np.repeat(users, NUM_DECISION_TIMES)
    data_dict['user_decision_t'] = np.stack([range(NUM_DECISION_TIMES) for _ in range(N)], axis=0).flatten()
    data_dict['trial_day_in_study'] = np.stack([1 + (np.arange(NUM_DECISION_TIMES) // 2) for _ in range(N)], axis=0).flatten()
    data_dict['day_in_study'] = np.stack([1 + (np.arange(NUM_DECISION_TIMES) // 2) for _ in range(N)], axis=0).flatten()
    for key in FILL_IN_COLS:
        data_dict[key] = np.full(batch_data_size, np.nan)
    data_df = pd.DataFrame.from_dict(data_dict)
    ### udpate df ###
    update_dict = {}
    num_updates = int(NUM_DECISION_TIMES / update_cadence)
    update_dict['user_idx'] = np.repeat(range(N), num_updates)
    update_dict['user_id'] = np.repeat(users, num_updates)
    update_dict['update_t'] = np.stack([np.arange(0, num_updates) for _ in range(N)], axis=0).flatten()
    for i in range(rl_algorithm_feature_dim):
        update_dict['posterior_mu.{}'.format(i)] = np.full(N * num_updates, np.nan)
    for i in range(rl_algorithm_feature_dim):
        for j in range(rl_algorithm_feature_dim):
            update_dict['posterior_var.{}.{}'.format(i, j)] = np.full(N * num_updates, np.nan)
    update_df = pd.DataFrame.from_dict(update_dict)

    return data_df, update_df

def create_dfs_full_pooling(user_ids, user_envs, rl_algorithm_feature_dim):
    N = len(user_ids)
    batch_data_size = N * NUM_DECISION_TIMES
    ### data df ###
    data_dict = {}
    data_dict['user_id'] = np.repeat(user_ids, NUM_DECISION_TIMES)
    data_dict['user_idx'] = np.repeat(list(user_envs.keys()), NUM_DECISION_TIMES)
    data_dict['user_decision_t'] = np.stack([range(NUM_DECISION_TIMES) for _ in range(N)], axis=0).flatten()
    data_dict['user_day_in_study'] = np.vectorize(dt_to_user_day_in_study)(data_dict['user_decision_t'])
    for key in FILL_IN_COLS:
        data_dict[key] = np.full(batch_data_size, np.nan)
    data_df = pd.DataFrame.from_dict(data_dict)
    ### udpate df ###
    update_dict = {}
    num_updates = 38 # the number of posterior updates in the real trial
    update_dict['update_t'] = np.arange(0, num_updates)
    for i in range(rl_algorithm_feature_dim):
        update_dict['posterior_mu.{}'.format(i)] = np.full(num_updates, np.nan)
    for i in range(rl_algorithm_feature_dim):
        for j in range(rl_algorithm_feature_dim):
            update_dict['posterior_var.{}.{}'.format(i, j)] = np.full(num_updates, np.nan)
    update_df = pd.DataFrame.from_dict(update_dict)

    return data_df, update_df

# regex pattern '.*' gets you everything
# Note: if regex pattern only refers to one column, then you need to .flatten() the resulting array
def get_data_df_values_for_users(data_df, user_idxs, trial_day_in_study, regex_pattern):
    return np.array(data_df.loc[(data_df['user_idx'].isin(user_idxs)) & (data_df['trial_day_in_study'] <= trial_day_in_study)].filter(regex=(regex_pattern)))

def get_user_data_values_from_decision_t(data_df, user_idx, decision_t, regex_pattern):
    return np.array(data_df.loc[(data_df['user_idx'] == user_idx) & (data_df['user_decision_t'] < decision_t)].filter(regex=(regex_pattern)))

def set_data_df_values_for_user(data_df, user_idx, decision_time, trial_day_in_study, policy_idx, action, prob, reward, quality, alg_state):
    data_df.loc[(data_df['user_idx'] == user_idx) & (data_df['user_decision_t'] == decision_time), FILL_IN_COLS] = np.concatenate([[trial_day_in_study, policy_idx, action, prob, reward, quality], alg_state])

### for full pooling experiments ###
def set_update_df_values(update_df, update_t, posterior_mu, posterior_var):
    update_df.iloc[update_df['update_t'] == update_t, 1:] = np.concatenate([posterior_mu, posterior_var.flatten()])

### for no pooling experiments ###
def set_update_df_values_for_user(update_df, user_idx, update_t, posterior_mu, posterior_var):
    update_df.iloc[(update_df['update_t'] == update_t) & (update_df['user_idx'] == user_idx), 3:] = np.concatenate([posterior_mu, posterior_var.flatten()])

# if user did not open the app at all before the decision time, then we simulate
# the algorithm selecting action based off of a stale state (i.e., b_bar is the b_bar from when the user last opened their app)
# if user did open the app, then the algorithm selecting action based off of a fresh state (i.e., b_bar stays the same)
def get_alg_state_from_app_opening(user_last_open_app_dt, data_df, user_idx, j, advantage_state):

    # if morning dt we check if users opened the app in the morning
    # if evening dt we check if users opened the app in the morning and in the evening
    if j % 2 == 0:
        user_opened_app_today = (user_last_open_app_dt == j)
    else:
        # we only simulate users opening the app for morning dts
        user_opened_app_today = (user_last_open_app_dt == j - 1)
    if not user_opened_app_today:
        # impute b_bar with stale b_bar and prior day app engagement = 0
        stale_b_bar = get_user_data_values_from_decision_t(data_df, user_idx, user_last_open_app_dt + 1, 'state.b.bar').flatten()[-1]
        # refer to rl_algorithm.py process_alg_state functions for V2, V3
        advantage_state[1] = stale_b_bar
        advantage_state[3] = 0

    return advantage_state

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

def execute_decision_time(data_df, user_idx, j, trial_day_in_study, alg_candidate, sim_env, policy_idx):
    env_state = sim_env.generate_current_state(user_idx, j)
    advantage_state, _ = alg_candidate.process_alg_state(env_state)
    # simulate app opening issue
    user_last_open_app_dt = sim_env.get_user_last_open_app_dt(user_idx)
    alg_state = get_alg_state_from_app_opening(user_last_open_app_dt, data_df, user_idx, j, advantage_state)
    ## ACTION SELECTION ##
    action, action_prob = alg_candidate.action_selection(alg_state)
    ## REWARD GENERATION ##
    # quality definition
    quality = sim_env.generate_outcomes(user_idx, env_state, action)
    # extract b_bar and a_bar from state
    b_bar = advantage_state[1]
    a_bar = advantage_state[2]
    reward = alg_candidate.reward_def_func(quality, action, b_bar, a_bar)
    ## SAVE VALUES ##
    set_data_df_values_for_user(data_df, user_idx, j, trial_day_in_study, policy_idx, action, action_prob, reward, quality, alg_state)

def run_experiment(alg_candidates, sim_env):
    env_users = sim_env.get_users()
    # all alg_candidates have the same update cadence and feature dimension
    update_cadence = alg_candidates[0].get_update_cadence()
    data_df, update_df = create_dfs_no_pooling(env_users, update_cadence, alg_candidates[0].get_feature_dim())
    policy_idxs = np.zeros(len(env_users))
    # add in prior values to posterior dataframe
    for user_idx in range(len(env_users)):
        set_update_df_values_for_user(update_df, user_idx, 0, \
        alg_candidates[user_idx].posterior_mean, alg_candidates[user_idx].posterior_var)
    for j in range(NUM_DECISION_TIMES):
        for user_idx in range(len(env_users)):
            alg_candidate = alg_candidates[user_idx]
            execute_decision_time(data_df, user_idx, j, dt_to_user_day_in_study(j), alg_candidate, sim_env, policy_idxs[user_idx])
            # each user's first week is pure exploration using the prior
            # note: we only update if the algorithm is online, or else the prior is used for the whole trial
            if ((j % update_cadence == (update_cadence - 1) and j >= 13) and alg_candidate.check_is_online()):
                day_in_study = 1 + (j // 2)
                alg_states = get_data_df_values_for_users(data_df, [user_idx], day_in_study, 'state.*')
                actions = get_data_df_values_for_users(data_df, [user_idx], day_in_study, 'action').flatten()
                pis = get_data_df_values_for_users(data_df, [user_idx], day_in_study, 'prob').flatten()
                rewards = get_data_df_values_for_users(data_df, [user_idx], day_in_study, 'reward').flatten()
                alg_candidate.update(alg_states, actions, pis, rewards)
                policy_idxs[user_idx] += 1
                update_idx = int(policy_idxs[user_idx])
                print("Update Time {} for {}".format(update_idx, user_idx))
                set_update_df_values_for_user(update_df, user_idx, update_idx, alg_candidate.posterior_mean, alg_candidate.posterior_var)

    return data_df, update_df

# either gets all users with that start date or end date
# type needs to be either "start" or "end"
def get_users_for_date(user_envs, date, type):
    users = []
    for user_idx, user_env in user_envs.items():
        if type == "start" and user_env.get_start_date() == date:
            users.append(user_idx)
        elif type == "end" and user_env.get_end_date() == date:
            users.append(user_idx)

    return users

### runs experiment with full pooling and incremental recruitment
def run_incremental_recruitment_exp(alg_candidate, sim_env):
    # instantiating dataframes
    env_users = sim_env.get_users()
    user_envs = sim_env.get_user_envs()
    data_df, update_df = create_dfs_full_pooling(env_users, user_envs, alg_candidate.get_feature_dim())
    # add in prior values to posterior dataframe
    update_idx = 0
    set_update_df_values(update_df, update_idx, alg_candidate.posterior_mean, alg_candidate.posterior_var)
    current_date_str, trial_end_date_str = sim_env.get_trial_start_end_dates()
    trial_day_in_study = 1
    # get current users
    current_user_idxs = get_users_for_date(user_envs, current_date_str, "start")
    while current_date_str != trial_end_date_str:
        # check if it's update time
        if ((get_date(current_date_str) in alg_candidate.get_update_dates()) and alg_candidate.check_is_online()):
            ### UPDATE TIME ###
            alg_states = get_data_df_values_for_users(data_df, current_user_idxs, trial_day_in_study, 'state.*')
            actions = get_data_df_values_for_users(data_df, current_user_idxs, trial_day_in_study, 'action').flatten()
            pis = get_data_df_values_for_users(data_df, current_user_idxs, trial_day_in_study, 'prob').flatten()
            rewards = get_data_df_values_for_users(data_df, current_user_idxs, trial_day_in_study, 'reward').flatten()
            alg_candidate.update(alg_states, actions, pis, rewards)
            update_idx += 1
            print(f"Update Time: {update_idx}")
            set_update_df_values(update_df, update_idx, alg_candidate.posterior_mean, alg_candidate.posterior_var)
        # execute morning and evening decision times for the current day
        for user_idx in current_user_idxs:
            user_start_date_str = user_envs[user_idx].get_start_date()
            user_day_in_study = calculate_day_in_study(user_start_date_str, current_date_str)
            morning_dt = get_morning_decision_t(user_day_in_study)
            evening_dt = get_evening_decision_t(user_day_in_study)
            execute_decision_time(data_df, user_idx, morning_dt, trial_day_in_study, alg_candidate, sim_env, update_idx)
            execute_decision_time(data_df, user_idx, evening_dt, trial_day_in_study, alg_candidate, sim_env, update_idx)

        # increment day
        current_date_str = increment_date(current_date_str)
        trial_day_in_study += 1
        # add users by start date
        current_user_idxs += get_users_for_date(user_envs, current_date_str, "start")
        # remove users if they have finished the trial
        finished_users = get_users_for_date(user_envs, current_date_str, "end")
        current_user_idxs = [user for user in current_user_idxs if user not in finished_users]

    return data_df, update_df
