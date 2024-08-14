import json
import os
import sys
import read_write_info
import run_experiments
import experiment_global_vars

# If this flag is set to True, the jobs won't be submitted to odyssey;
# they will instead be ran one after another in your current terminal
# session. You can use this to either run a sequence of jobs locally
# on your machine, or to run a sequence of jobs one after another
# in an interactive shell on odyssey.
DRYRUN = True
JOB_TYPE = "simulations" #simulations #compute_metrics
# In this package, there are two types of simulations:
# (1) Evaluation: re-evaluate algorithm design decisions of the Oralytics RL algorithm
# (2) Did We Learn? assess if the algorithm learned: namely there's evidence that the RL algorithm 
# learend that's not due to the stochastic action-selection

# This is the base directory where the results will be stored.
OUTPUT_DIR = read_write_info.WRITE_PATH_PREFIX

# This list contains the jobs and simulation enviornments and algorithm
# candidates to search over.
# The list consists of tuples, in which the first element is
# the name of the job (here it describes the exp we want to run)
# and the second is a dictionary of parameters that will be
# be grid-searched over.
# Note that the second parameter must be a dictionary in which each
# value is a list of options.
CLUSTER_SIZES = ["full_pooling"]
OFFLINE_OR_ONLINE = ["online"]
SEEDS = range(experiment_global_vars.MAX_SEED_VAL)
# Notice: if you do not specify state featuers then you are running "evaluation" simulations
# otherwise, you are running "did we learn?"" simulations
STATE_FEATURES = ["bias"] # Note: "bias" signifies the variant where there is no advantage in ANY state
# Namely, we zero out all of the advantage parameter vectors
# STATE_FEATURES = ["tod", "b_bar", "a_bar", "app_engage", "bias"] 

### RUNNING EVALUATIONS ###
# QUEUE = [
#     ('eval_online', dict(
#                     cluster_size=["full_pooling"],
#                     offline_or_online=OFFLINE_OR_ONLINE,
#                     state_feature=[None],
#                     seed=SEEDS
#                     )
#     ),
#     ('eval_pooling', dict(
#                        cluster_size=CLUSTER_SIZES,
#                        offline_or_online=["online"],
#                        state_feature=[None],
#                        seed=SEEDS
#                        )
#     )
#     ]

### RUNNING DID WE LEARN? ###
QUEUE = [
    ('did_we_learn', dict(
                    cluster_size=["full_pooling"],
                    offline_or_online=["online"],
                    state_feature=STATE_FEATURES,
                    seed=SEEDS
                    )
    )
    ]

def run(exp_dir, exp_name, exp_kwargs):
    '''
    This is the function that will actually execute the job.
    To use it, here's what you need to do:
    1. Create directory 'exp_dir' as a function of 'exp_kwarg'.
       This is so that each set of experiment+hyperparameters get their own directory.
    '''
    exp_path = os.path.join(exp_dir, f'{exp_name}_{exp_kwargs["state_feature"]}')
    print('Results will be stored stored in:', exp_path)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    '''
    2. Run your experiment
    Note: Results are saved after every seed in run_experiments
    '''
    print('Running experiment {}:'.format(exp_name))
    run_experiments.run_experiment(exp_kwargs, exp_path, JOB_TYPE)
    '''
    3. You can find results in 'exp_dir'
    '''
    print('Results are stored in:', exp_path)
    print('with experiment parameters', exp_kwargs)
    print('\n')


def main():
    assert(len(sys.argv) > 2)

    exp_dir = sys.argv[1]
    exp_name = sys.argv[2]
    exp_kwargs = json.loads(sys.argv[3])

    run(exp_dir, exp_name, exp_kwargs)


if __name__ == '__main__':
    main()
