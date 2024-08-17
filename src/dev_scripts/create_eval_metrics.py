
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

NUM_SIMULATIONS = 500

def get_proportion_missed_df(qualities):
  df = qualities.groupby('user_id').apply(lambda x: (x == 0).mean()).drop(columns='user_id').reset_index()
  metrics = df.iloc[:,2:].mean(axis=0)

  return metrics, "Proportion of Decision Times With OSCB = 0"

def get_avg_nz_brushing_duration_unpooled_df(qualities):
  df = qualities.groupby('user_id').apply(lambda x: x.iloc[:, 1:][x.iloc[:, 1:] != 0].mean()).reset_index()
  metrics = df.iloc[:,2:].mean(axis=0)

  return metrics, "Average of Average Non-Zero Participant OSCB"

def get_avg_nz_brushing_duration_pooled_df(qualities):
  def get_avg_nz_qual(df, col):
    return df[df[col] != 0][col].mean()
  def get_std_nz_qual(df, col):
      return df[df[col] != 0][col].std()
  df = qualities.iloc[:,2:].apply(lambda col: get_avg_nz_qual(qualities, col.name))
  metrics = df

  return metrics, "Average Non-Zero OSCB in Trial"

def get_variance_nz_brushing_duration_unpooled_df(qualities):
  df = qualities.groupby('user_id').apply(lambda x: x.iloc[:, 1:][x.iloc[:, 1:] != 0].var()).reset_index()
  metrics = df.iloc[:,2:].mean(axis=0)

  return metrics, "Variance of Average Non-Zero Participant OSCB"

def get_variance_nz_brushing_duration_pooled_df(qualities):
  def get_var_nz_qual(df, col):
      return df[df[col] != 0][col].var()

  df = qualities.drop(columns=['user_id']).apply(lambda col: get_var_nz_qual(qualities, col.name)).iloc[1:]
  metrics = df

  return metrics, "Variance of Non-Zero OSCB in Trial"

def get_variance_avg_user_brushing_duration_df(qualities):
  df = qualities.groupby('user_id').mean().reset_index()
  metrics = df.iloc[:,2:].mean(axis=0)

  return metrics, "Variance of Average Participant OSCB"

def get_avg_variances_within_user_brushing_duration_df(qualities):
  df = qualities.groupby('user_id').var().reset_index()
  metrics = df.iloc[:,2:].mean(axis=0)

  return metrics, "Average of Variances of Participant OSCB"

def make_metrics_comparison_table_row(metrics_df, metric_name):
  mean_sim_val = metrics_df.iloc[1:].mean()
  se_sim_val = metrics_df.iloc[1:].std()/np.sqrt(NUM_SIMULATIONS)
  true_val = metrics_df.iloc[0]
  sim_val_text = mean_sim_val.round(3).astype(str) + ' (' + se_sim_val.round(3).astype(str) + ')'

  table_row = pd.DataFrame({'Metric': [metric_name], 'Simulation Environment': [sim_val_text], 'Oralytics Trial Data': [true_val]})
  return table_row

def make_metrics_comparison_table(qualities):
  metrics_df, metric_name = get_proportion_missed_df(qualities)
  table_row_1 = make_metrics_comparison_table_row(metrics_df, metric_name)

  metrics_df, metric_name = get_avg_nz_brushing_duration_pooled_df(qualities)
  table_row_2 = make_metrics_comparison_table_row(metrics_df, metric_name)

  metrics_df, metric_name = get_avg_nz_brushing_duration_unpooled_df(qualities)
  table_row_3 = make_metrics_comparison_table_row(metrics_df, metric_name)

  metrics_df, metric_name = get_variance_nz_brushing_duration_pooled_df(qualities)
  table_row_4 = make_metrics_comparison_table_row(metrics_df, metric_name)

  metrics_df, metric_name = get_variance_nz_brushing_duration_unpooled_df(qualities)
  table_row_5 = make_metrics_comparison_table_row(metrics_df, metric_name)

  metrics_df, metric_name = get_variance_avg_user_brushing_duration_df(qualities)
  table_row_6 = make_metrics_comparison_table_row(metrics_df, metric_name)

  metrics_df, metric_name = get_avg_variances_within_user_brushing_duration_df(qualities)
  table_row_7 = make_metrics_comparison_table_row(metrics_df, metric_name)

  return pd.concat([table_row_1, table_row_2, table_row_3, table_row_4, table_row_5, table_row_6, table_row_7], axis=0)

def plot_ci_pooled_by_user(data, sim_qual_cols, ax, title=""):
    data['sim_mean'] = data[sim_qual_cols].mean(axis=1)
    stds = data[sim_qual_cols].std(axis=1)
    confidence_level = 0.95

    z_score = np.abs(norm.ppf((1 - confidence_level) / 2))

    margin_of_error = z_score * stds

    data['lower_bound'] = data['sim_mean'] - margin_of_error
    data['upper_bound'] = data['sim_mean'] + margin_of_error

    # Check if the true value is within the prediction interval for each row
    within_interval = (data['quality'] >= data['lower_bound']) & (data['quality'] <= data['upper_bound'])

    data.loc[:, 'within_interval'] = within_interval.astype(int)
    data = data.drop(columns=sim_qual_cols)

    user_ids = data['user_id'].str.extract(r'\+(\d+)').astype(str)[0]
    true_means = data['quality']
    sim_means = data['sim_mean']
    lower_bounds = data['lower_bound']
    upper_bounds = data['upper_bound']

    ax.scatter(range(len(user_ids)), true_means, color='blue', label='True Mean')
    ax.scatter(range(len(user_ids)), sim_means, color='orange', label='Simulated Mean')

    for i in range(len(user_ids)):
        ax.errorbar(i, sim_means[i], yerr=[[sim_means[i] - lower_bounds[i]], [upper_bounds[i] - sim_means[i]]],
                    fmt='none', ecolor='black', capsize=5)

    ax.set_xlabel('User ID', fontsize=16)
    ax.set_ylabel('Mean', fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(range(len(user_ids)))
    ax.set_xticklabels(user_ids, rotation=90)  # Rotate labels if needed for better readability

    ax.legend()

def plot_ci_all_features(qualities, sim_qual_cols):
  avg_nz_qual = qualities.groupby('user_id').apply(lambda x: x.iloc[:, 1:][x.iloc[:, 1:] != 0].mean()).reset_index()
  var_nz_qual = qualities.groupby('user_id').apply(lambda x: x.iloc[:, 1:][x.iloc[:, 1:] != 0].var()).reset_index()
  prop_missed = qualities.groupby('user_id').apply(lambda x: (x == 0).mean()).drop(columns='user_id').reset_index()

  fig, axs = plt.subplots(3, 1, figsize=(15, 15))

  plot_ci_pooled_by_user(prop_missed, sim_qual_cols, axs[1], "Proportion of Decision Times with OSCB = 0")
  plot_ci_pooled_by_user(avg_nz_qual, sim_qual_cols, axs[0], "Average Non-Zero Participant OSCB")
  plot_ci_pooled_by_user(var_nz_qual, sim_qual_cols, axs[2], "Variance of Non-zero Participant OSCB")

  plt.tight_layout()
  plt.savefig('../../figs/simulation_metrics_plot.png')

def make_error_values_table(qualities,sim_qual_cols):
    differences = qualities[sim_qual_cols].sub(qualities['quality'], axis=0)
    
    mse_values = (differences**2).mean(axis=0)
    mse = mse_values.mean()
    mse_se = mse_values.std() / np.sqrt(len(mse_values))
    
    rmse_values = mse_values**0.5
    rmse = rmse_values.mean()
    rmse_se = rmse_values.std() / np.sqrt(len(rmse_values))
    
    mae_values = np.abs(differences).mean(axis=0)
    mae = mae_values.mean()
    mae_se = mae_values.std() / np.sqrt(len(mae_values))
    
    table_rows = pd.DataFrame({
        'Metrics': ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error'], 
        'Value': [
            f"{mse:.3f} ({mse_se:.3f})", 
            f"{rmse:.3f} ({rmse_se:.3f})", 
            f"{mae:.3f} ({mae_se:.3f})"
        ]
    })
    
    return table_rows
