import os
import pickle
import pandas as pd

def read_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

directories = {
    '../eval_online/eval_online_full_pooling_offline': ('Full Pooling', 'Offline'),
    '../eval_online/eval_online_full_pooling_online': ('Full Pooling', 'Online'),
    '../eval_pooling/eval_pooling_full_pooling_online': ('Full Pooling', 'Online'),
    '../eval_pooling/eval_pooling_no_pooling_online': ('No Pooling', 'Online')
}

data = []
for directory, (pooling_type, online_offline) in directories.items():
    print('FILE', os.path.join(directory, 'avg.p'))

    avg_value = read_pickle(os.path.join(directory, 'avg.p'))
    low_25_value = read_pickle(os.path.join(directory, 'low_25.p'))
    data.append({
        'Pooling': pooling_type,
        'Online vs. Offline': online_offline,
        'Mean Value': avg_value,
        'Low 25th Percentile': low_25_value
    })

df = pd.DataFrame(data)
latex_table = df.to_latex(index=False, caption='Evaluation Results', label='tab:evaluation_results')
with open('results_table.tex', 'w') as file:
    file.write(latex_table)

