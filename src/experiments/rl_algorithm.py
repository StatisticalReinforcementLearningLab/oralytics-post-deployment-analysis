# -*- coding: utf-8 -*-
"""
RL Algorithm that uses a contextual bandit framework with Thompson sampling, full-pooling, and
a Bayesian Linear Regression reward approximating function.
"""
import reward_definition
import smoothing_function

import numpy as np
import scipy.stats as stats
from datetime import datetime, timedelta

class RLAlgorithm():
    def __init__(self, cost_params, update_cadence, smoothing_func):
        # how often the RL algorithm updates parameters
        self.update_cadence = update_cadence
        # smoothing function for after-study analysis
        self.smoothing_func = smoothing_func
        # feature space dimension
        self.feature_dim = 0
        # xi_1, xi_2 params for the cost term parameterizes the reward def. func.
        self.reward_def_func = lambda brushing_quality, current_action, b_bar, a_bar: \
                      reward_definition.calculate_reward(brushing_quality, \
                                        cost_params[0], cost_params[1], \
                                        current_action, b_bar, a_bar)

    # need to implement
    # function that takes in a raw state and processes the current state for the algorithm
    def process_alg_state(self, env_state, b_bar, a_bar):
        return 0

    def action_selection(self, advantage_state, baseline_state):
        return 0

    def update(self, alg_states, actions, pis, rewards):
        return 0

    def get_feature_dim(self):
        return self.feature_dim

    def get_update_cadence(self):
        return self.update_cadence

"""### Bayesian Linear Regression Thompson Sampler
---
"""
## POSTERIOR HELPERS ##
# create the feature vector given state, action, and action selection probability

# with action centering
def create_big_phi(advantage_states, baseline_states, actions, probs):
  big_phi = np.hstack((baseline_states, np.multiply(advantage_states.T, probs).T, \
                       np.multiply(advantage_states.T, (actions - probs)).T,))
  return big_phi

"""
#### Helper Functions
---
"""

def compute_posterior_var(Phi, sigma_n_squared, prior_sigma):
  return np.linalg.inv(1/sigma_n_squared * Phi.T @ Phi + np.linalg.inv(prior_sigma))

def compute_posterior_mean(Phi, R, sigma_n_squared, prior_mu, prior_sigma):

  return compute_posterior_var(Phi, sigma_n_squared, prior_sigma) \
   @ (1/sigma_n_squared * Phi.T @ R + np.linalg.inv(prior_sigma) @ prior_mu)

# update posterior distribution
def update_posterior_w(Phi, R, sigma_n_squared, prior_mu, prior_sigma):
  mean = compute_posterior_mean(Phi, R, sigma_n_squared, prior_mu, prior_sigma)
  var = compute_posterior_var(Phi, sigma_n_squared, prior_sigma)

  return mean, var

## ACTION SELECTION ##
# we calculate the posterior probability of P(R_1 > R_0) clipped
# we make a Bernoulli draw with prob. P(R_1 > R_0) of the action
def bayes_lr_action_selector(beta_post_mean, beta_post_var, advantage_state, smoothing_func):
  # using the genearlized_logistic_func, probabilities are already clipped to asymptotes
  mu = advantage_state @ beta_post_mean
  std = np.sqrt(advantage_state @ beta_post_var @ advantage_state.T)
  posterior_prob = stats.norm.expect(func=smoothing_func, loc=mu, scale=std)

  return stats.bernoulli.rvs(posterior_prob), posterior_prob

# algorithm candidates that run in the V4 environment
# uses the prior built and deployed in the Oralyitcs MRT
class OralyticsMRTAlg(RLAlgorithm):
    def __init__(self, offline_or_online):
        self.is_online = True if offline_or_online == "online" else False
        cost_params = [100, 100]
        update_cadence = 14 # denotes weekly updates, this value is only used by no-pooling algorithms
        smoothing_func = smoothing_function.stable_generalized_logistic
        super(OralyticsMRTAlg, self).__init__(cost_params, update_cadence, smoothing_func)

        # size of mu vector = D_baseline=5 + D_advantage=5 + D_advantage=5
        self.D_ADVANTAGE = 5
        self.D_BASELINE = 5
        self.feature_dim = self.D_BASELINE + self.D_ADVANTAGE + self.D_ADVANTAGE

        ALPHA_0_MU = [18, 0, 30, 0, 73]
        BETA_MU = [0, 0, 0, 53, 0]
        self.PRIOR_MU = np.array(ALPHA_0_MU + BETA_MU + BETA_MU)
        ALPHA_0_SIGMA = [73**2, 25**2, 95**2, 27**2, 83**2]
        BETA_SIGMA = [12**2, 33**2, 35**2, 56**2, 17**2]
        self.PRIOR_SIGMA = np.diag(np.array(ALPHA_0_SIGMA + BETA_SIGMA + BETA_SIGMA))
        self.posterior_mean = np.copy(self.PRIOR_MU)
        self.posterior_var = np.copy(self.PRIOR_SIGMA)

        self.SIGMA_N_2 = 3878

        # generating update times as in the real Oralyitcs trial
        update_start_date = datetime.strptime("2023-11-05", "%Y-%m-%d").date()
        update_end_date = datetime.strptime("2024-07-14", "%Y-%m-%d").date()
        self.update_dates = [update_start_date + timedelta(weeks=i) for i in range((update_end_date - update_start_date).days // 7 + 1) if update_start_date + timedelta(weeks=i) <= update_end_date]

    """
    Note: In V4, the environment state already calculates a normalized version of b_bar and a_bar.
    Please refer to generate_env_state in sim_env_v4

    Algorithm State Space Used In Oralytics MRT
    # 
    ## baseline: ##
    # 0 - time of day
    # 1 - b bar
    # 2 - a bar
    # 3 - app engagement
    # 4 - bias
    ## advantage: ##
    # 0 - time of day
    # 1 - b bar
    # 2 - a bar
    # 3 - app engagement
    # 4 - bias
    """
    def process_alg_state(self, env_state):
        baseline_state = np.array([env_state[0], env_state[1], \
                                env_state[2], env_state[3], 1])
        advantage_state = np.copy(baseline_state)

        return advantage_state, baseline_state

    def action_selection(self, advantage_state):
        return bayes_lr_action_selector(self.posterior_mean[-self.D_ADVANTAGE:], \
                                            self.posterior_var[-self.D_ADVANTAGE:,-self.D_ADVANTAGE:], \
                                            advantage_state, \
                                            self.smoothing_func)
    
    def check_is_online(self):
        return self.is_online
    
    def get_update_dates(self):
        return self.update_dates

    def update(self, alg_states, actions, pis, rewards):
        Phi = create_big_phi(alg_states, alg_states, actions, pis)
        posterior_mean, posterior_var = update_posterior_w(Phi, rewards, self.SIGMA_N_2, self.PRIOR_MU, self.PRIOR_SIGMA)
        self.posterior_mean = posterior_mean
        self.posterior_var = posterior_var
