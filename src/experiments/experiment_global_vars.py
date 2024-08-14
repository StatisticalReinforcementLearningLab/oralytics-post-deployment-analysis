"""
GLOBAL VALUES
"""

MAX_SEED_VAL = 500
NUM_TRIALS = 70
TRIAL_LENGTH_IN_WEEKS = 10
# We should have NUM_USERS x NUM_DECISION_TIMES datapoints for each saved value or
# statistic at the end of the study
NUM_DECISION_TIMES = 70 * 2
NUM_ALG_UPDATES = 38

"""
V4:
* recruitment rate is based off of the exact start date from the MRT
* app engagement in algorithm state
"""
FILL_IN_COLS = ['trial_day_in_study', 'policy_idx', 'action', 'prob', 'reward', 'quality'] + ['state.tod', 'state.b.bar',\
 'state.a.bar', 'state.app.engage', 'state.bias']
