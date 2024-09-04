# oralytics-post-deployment-analysis
This repository contains code from the [Oralytics Deployment Paper](https://arxiv.org/abs/2409.02069) for performing re-sampling analyses to re-evaluate algorithm decisions made for the RL algorithm deployed in the MRT (phase 1) of the Oralytics trial.

## Citing Our Code
If you use our code in any way, please cite us:
```
@misc{trella2024deployedonlinereinforcementlearning,
      title={A Deployed Online Reinforcement Learning Algorithm In An Oral Health Clinical Trial}, 
      author={Anna L. Trella and Kelly W. Zhang and Hinal Jajal and Inbal Nahum-Shani and Vivek Shetty and Finale Doshi-Velez and Susan A. Murphy},
      year={2024},
      eprint={2409.02069},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2409.02069}, 
}
```

## Fitting Simulation Environment
* Running `python3 src/dev_scripts/fitting_user_models.py` will fit each Oralytics participant to a non-stationary base model class (zero-inflated poisson model) and save parameters to `v4_non_stat_zip_model_params.csv`.
* Running `python3 src/dev_scripts/app_opening_prob_calculation.py.py` will fit an app opening probability for each Oralytics participant and save the probabilities to `v4_app_open_prob.csv`.
* Running `python3 src/dev_scripts/get_participant_start_end_dates.py` will get the start and end dates (i.e., when the participant started and completed the trial) of each Oralytics participant and save the info. to `v4_start_end_dates.csv`.

## Evaluating Simulation Environment
* Running `python3 src/dev_scripts/eval_sim_env.py` calculates various metrics comparing the data generating by the fitted simulation environment with data from the Oralytics trial.

## Running Re-Sampling Based Experiments
Experiments can be run sequentially one at a time or in parallel. In this paper, we ran two types of experiments (1) re-evaluate design decisions made for the Oralytics algorithm and (2) investigate what the algorithm learned (did we learn?)

To run experiments:

1. Fill in the read and write path in `read_write_info.py`. This specifies what path to read data from and what path to write results to.
2. In `run.py`, specify experiment parameters as instructed in the file. Example: specify the simulation environment variants and algorithm candidate properties. You must have the field value `JOB_TYPE = "simulations"` to run these experiments. In addition, you must modify the DRYRUN field to specify running jobs in parallel or sequentially. DRYRUN = True runs jobs one after the other (this is a good practice to test out new code initially). Switch to DRYRUN = False to run experiments in parallel.

The experiment parameters one can specify in `run.py` are:
a. `cluster_size = ["full_pooling", "no_pooling"]` (full-pooling vs. no-pooling algorithm)
b. `offline_or_online = ["online", "offline"]` (online algorithm that updates as data accrues vs. offline algorithm that only uses the prior and does not update the policy)
c. `seed = range(MAX_SEED_VAL)` (`MAX_SEED_VAL` specifies how many Monte-Carlo repititions you want to run)
d. `state` (if `state=[None]` then you are running experiment type (1) for re-evaluating design decisions, otherwise, you are running experiment type (2) did we learn? and `state` needs to be a list with 4 elements corresponding to state feature values that together form the state of interest)

3. Run `python3 src/experiments/submit_batch.py` on the cluster to submit jobs and run in parallel.

## Computing Metrics and Plotting Figures
* For experiment type (1) re-evaluating design decisions, we calculate various metrics on the re-sampled outcomes. To calculate these metrics after running experiments in the above section, first change the `JOB_TYPE` field in `run.py` to `JOB_TYPE = "compute_metrics"` and then run `python3 src/experiments/submit_batch.py` 
* For experiment type (2) did we learn?, we visualize the standardized predicted advantage in state s throughout the trial. To plot these visualization using results after running experiments in the above section, run `python3 src/experiments/plot_predicted_advs.py` 
