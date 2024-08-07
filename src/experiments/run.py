import json
import os
import sys
import read_write_info
import run_experiments

# If this flag is set to True, the jobs won't be submitted to odyssey;
# they will instead be ran one after another in your current terminal
# session. You can use this to either run a sequence of jobs locally
# on your machine, or to run a sequence of jobs one after another
# in an interactive shell on odyssey.
DRYRUN = True
JOB_TYPE = "simulations" #simulations #compute_metrics

# This is the base directory where the results will be stored.
# On Odyssey, you may not want this to be your home directory
# If you're storing lots of files (or storing a lot of data).
OUTPUT_DIR = read_write_info.WRITE_PATH_PREFIX

# This list contains the jobs and simulation enviornments and algorithm
# candidates to search over.
# The list consists of tuples, in which the first element is
# the name of the job (here it describes the exp we want to run)
# and the second is a dictionary of parameters that will be
# be grid-searched over.
# Note that the second parameter must be a dictionary in which each
# value is a list of options.
CLUSTER_SIZES = ["full_pooling", "no_pooling"]
OFFLINE_OR_ONLINE = ["offline", "online"]

### RUNNING SIMULATIONS ###
QUEUE = [
    ('eval_online', dict(
                    cluster_size=["full_pooling"],
                    offline_or_online=OFFLINE_OR_ONLINE
                    )
    ),
    ('eval_pooling', dict(
                       cluster_size=CLUSTER_SIZES,
                       offline_or_online=["online"]
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
    exp_path = os.path.join(exp_dir, exp_name)
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
