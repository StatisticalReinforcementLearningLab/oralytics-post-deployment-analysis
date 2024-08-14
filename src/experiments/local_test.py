import sim_env_v4
import rl_experiments
import rl_algorithm
import copy
import numpy as np

# np.random.seed(0)
# environment_module = sim_env_v4.SimulationEnvironmentV4()
# alg_candidate = rl_algorithm.OralyticsMRTAlg("online")
# data_df, update_df = rl_experiments.run_incremental_recruitment_exp(alg_candidate, environment_module)
# data_df.to_csv('test_data_df_online_pooling.csv', index=False)
# update_df.to_csv('test_update_df_online_pooling.csv', index=False)

# np.random.seed(0)
# environment_module = sim_env_v4.SimulationEnvironmentV4()
# alg_candidate = rl_algorithm.OralyticsMRTAlg("offline")
# data_df, update_df = rl_experiments.run_incremental_recruitment_exp(alg_candidate, environment_module)
# data_df.to_csv('test_data_df_offline_pooling.csv', index=False)
# update_df.to_csv('test_update_df_offline_pooling.csv', index=False)

np.random.seed(0)
environment_module = sim_env_v4.SimulationEnvironmentV4()
alg_candidate = rl_algorithm.OralyticsMRTAlg("online")
alg_candidates = [copy.deepcopy(alg_candidate) for _ in range(79)]
data_df, update_df = rl_experiments.run_experiment(alg_candidates, environment_module)
data_df.to_csv('test_data_df_online_no_pooling.csv', index=False)
update_df.to_csv('test_update_df_online_no_pooling.csv', index=False)