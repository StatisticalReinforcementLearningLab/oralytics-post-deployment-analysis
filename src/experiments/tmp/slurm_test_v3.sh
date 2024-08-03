#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared           # Partition to submit to
#SBATCH --mem=5000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /n/holyscratch01/murphy_lab/Users/hjajal/test_2/test_v3/out_%j.txt # File to which STDOUT will be written
#SBATCH -e /n/holyscratch01/murphy_lab/Users/hjajal/test_2/test_v3/err_%j.txt # File to which STDERR will be written

python -u run.py /n/holyscratch01/murphy_lab/Users/hjajal/test_2/test_v3 test_v3 '{"sim_env_version": "v3", "base_env_type": "STAT", "effect_size_scale": "None", "delayed_effect_scale": "HIGH_R", "alg_type": "BLR_AC_V3", "noise_var": "None", "clipping_vals": [0.2, 0.8], "b_logistic": 0.515, "update_cadence": 14, "cluster_size": "no_pooling", "cost_params": [180, 180]}'
