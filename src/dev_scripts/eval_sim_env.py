
import pandas as pd
import numpy as np
from scipy.stats import bernoulli
from scipy.stats import poisson
from scipy.stats import norm
import matplotlib.pyplot as plt

from fitting_user_models import process_mrt_data

NUM_SIMULATIONS = 500
MRT_DATA = pd.read_csv('oralytics_mrt_data.csv')
MRT_DATA = process_mrt_data(MRT_DATA)
NON_STAT_PARAMS_DF = pd.read_csv('../../sim_env_data/v4_non_stat_zip_model_params.csv')
STATE_COLS = [column for column in MRT_DATA.columns if 'state_' in column]


# note: since v4 only chose zip models, these are the following parameters
def get_base_params_for_user(user):
  param_dim = 7
  param_df = NON_STAT_PARAMS_DF
  user_row = np.array(param_df[param_df['User'] == user])
  bern_base = user_row[0][2:2 + param_dim]
  poisson_base = user_row[0][2 + 2 * param_dim:2 + 3 * param_dim]

  # poisson parameters, bernouilli parameters
  return bern_base, poisson_base, None

# note: since v4 only chose zip models, these are the following parameters
def get_adv_params_for_user(user):
  param_dim = 7
  param_df = NON_STAT_PARAMS_DF
  user_row = np.array(param_df[param_df['User'] == user])
  bern_adv = user_row[0][2 + param_dim: 2 + 2 * param_dim]
  poisson_adv = user_row[0][2 + 3 * param_dim:2 + 4 * param_dim]

  return bern_adv, poisson_adv


def get_user_effect_funcs():
    # negative treatment effect means users are more likely to brush
    bern_adv_func = lambda state, adv_params: min(adv_params @ state, 0)
    # positive treatment effect means users brush more seconds
    y_adv_func = lambda state, adv_params: max(adv_params @ state, 0)

    return bern_adv_func, y_adv_func

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def construct_model_and_sample(state, action, \
                                          bern_base_params, \
                                          y_base_params, \
                                          bern_adv_params, \
                                          y_adv_params, \
                                          sigma_u, \
                                          model_type, \
                                          effect_func_bern=lambda state : 0, \
                                          effect_func_y=lambda state : 0):
  bern_linear_comp = state @ bern_base_params
  if (action == 1):
    bern_linear_comp += effect_func_bern(state, bern_adv_params)
  bern_p = 1 - sigmoid(bern_linear_comp)
  # bernoulli component
  rv = bernoulli.rvs(bern_p)
  if (rv):
      y_mu = state @ y_base_params
      if (action == 1):
          y_mu += effect_func_y(state, y_adv_params)
      if model_type == "sqrt_norm":
        # normal transform component
        sample = norm.rvs(loc=y_mu, scale=sigma_u)
        sample = sample**2

        # we round to the nearest integer to produce brushing duration in seconds
        return int(sample)
      else:
        # poisson component
        l = np.exp(y_mu)
        sample = poisson.rvs(l)*rv

        return sample

  else:
    return 0

for idx, row in MRT_DATA.iterrows():
    user = row['user_id']
    state = np.array(row[STATE_COLS])
    action = row['action']

    # Retrieve the parameters for the user
    bern_base_params, y_base_params, _ = get_base_params_for_user(user)
    bern_adv_params, y_adv_params = get_adv_params_for_user(user)
    effect_func_bern, effect_func_y = get_user_effect_funcs()

    # Run simulations and store the results in new columns
    for i in range(NUM_SIMULATIONS):
        quality = construct_model_and_sample(state, action,
                                             bern_base_params,
                                             y_base_params,
                                             bern_adv_params,
                                             y_adv_params,
                                             None,
                                             'zip',
                                             effect_func_bern,
                                             effect_func_y)
        sim_quality_col = f'sim_quality_{i+1}'
        MRT_DATA.at[idx, sim_quality_col] = quality

sim_quality = MRT_DATA.filter(regex='^sim_quality|^user_id')
MRT_DATA['mean_sim_quality'] = sim_quality.drop(columns='user_id').mean(axis=1)

sim_qual_cols = [column for column in MRT_DATA.columns if column.startswith('sim_quality_') ]
qualities = MRT_DATA[['user_id','quality']+sim_qual_cols]

class SimulationMetrics:
    def __init__(self, qualities, sim_qual_cols=None):
        self.qualities = qualities
        self.sim_qual_cols = sim_qual_cols

    def get_proportion_missed(self):
        prop_missed = self.qualities.groupby('user_id').apply(lambda x: (x == 0).mean()).drop(columns='user_id').reset_index()
        if self.sim_qual_cols:
          prop_missed['Simulations'] = prop_missed.iloc[:, 2:].mean(axis=1)
          prop_missed = prop_missed.drop(columns=self.sim_qual_cols)
        prop_missed = prop_missed.rename(columns={'quality':'Pilot'})
        prop_missed_values = prop_missed.iloc[:, 1:].mean()
        prop_missed_values = pd.DataFrame(prop_missed_values, columns=['Proportion of Missed Brushing Windows']).reset_index()
        table_row = pd.pivot_table(prop_missed_values, columns='index', values='Proportion of Missed Brushing Windows', aggfunc='first')
        table_row = table_row.reset_index().rename(columns={'index': 'Metrics'})
        return table_row

    def get_avg_nz_brushing_duration_pooled(self):
        avg_nz_qual = self.qualities.groupby('user_id').apply(lambda x: x.iloc[:, 1:][x.iloc[:, 1:] != 0].mean()).reset_index()
        if self.sim_qual_cols:
          avg_nz_qual['Simulations'] = avg_nz_qual.iloc[:, 2:].mean(axis=1)
          avg_nz_qual = avg_nz_qual.drop(columns=self.sim_qual_cols)
        avg_nz_qual = avg_nz_qual.rename(columns={'quality':'Pilot'})
        avg_nz_qual_values = avg_nz_qual.iloc[:, 1:].mean()
        avg_nz_qual_values = pd.DataFrame(avg_nz_qual_values, columns=['Average Non-Zero BDs - Pooled by User']).reset_index()
        table_row = pd.pivot_table(avg_nz_qual_values, columns='index', values='Average Non-Zero BDs - Pooled by User', aggfunc='first')
        table_row = table_row.reset_index().rename(columns={'index': 'Metrics'})
        return table_row

    def get_avg_nz_brushing_duration_unpooled(self):
        def get_avg_nz_qual(df, col):
            return df[df[col] != 0][col].mean()
        true_val = self.qualities.drop(columns=['user_id']).apply(lambda col: get_avg_nz_qual(self.qualities, col.name))['quality']
        if self.sim_qual_cols:
          mean_sim_val = self.qualities.drop(columns=['user_id']).apply(lambda col: get_avg_nz_qual(self.qualities, col.name)).drop('quality').mean()
          table_row = pd.DataFrame({'Metrics': ['Average Non-Zero BDs - Unpooled'], 'Simulations': [mean_sim_val], 'Pilot': [true_val]})
        else:
          table_row = pd.DataFrame({'Metrics': ['Average Non-Zero BDs - Unpooled'], 'Pilot': [true_val]})
        return table_row

    def get_variance_nz_brushing_duration_pooled(self):
        var_nz_qual = self.qualities.groupby('user_id').apply(lambda x: x.iloc[:, 1:][x.iloc[:, 1:] != 0].var()).reset_index()
        if self.sim_qual_cols:
          var_nz_qual['Simulations'] = var_nz_qual[self.sim_qual_cols].mean(axis=1)
          var_nz_qual = var_nz_qual.drop(columns=self.sim_qual_cols)
        var_nz_qual = var_nz_qual.rename(columns={'quality':'MRT Data'})
        var_nz_qual_values = var_nz_qual.iloc[:, 1:].mean()
        var_nz_qual_values = pd.DataFrame(var_nz_qual_values, columns=['Variance of Non-Zero BDs - Pooled by User']).reset_index()
        table_row = pd.pivot_table(var_nz_qual_values, columns='index', values='Variance of Non-Zero BDs - Pooled by User', aggfunc='first')
        table_row = table_row.reset_index().rename(columns={'index': 'Metrics'})
        return table_row

    def get_variance_nz_brushing_duration_unpooled(self):
        def get_var_nz_qual(df, col):
            return df[df[col] != 0][col].var()
        var_nz_qual = self.qualities.drop(columns=['user_id']).apply(lambda col: get_var_nz_qual(self.qualities, col.name))
        true_val = var_nz_qual['quality']
        if self.sim_qual_cols:
          mean_sim_val = var_nz_qual[self.sim_qual_cols].mean()
          table_row = pd.DataFrame({'Metrics': ['Variance of Non-Zero BDs - Unpooled'], 'Simulations': [mean_sim_val], 'MRT Data': [true_val]})
        else:
          table_row = pd.DataFrame({'Metrics': ['Variance of Non-Zero BDs - Unpooled'], 'MRT Data': [true_val]})
        return table_row

    def get_variance_avg_user_brushing_duration(self):
        avg_user_qual = self.qualities.groupby('user_id').mean().reset_index()
        var_avg_user_qual = avg_user_qual.iloc[:, 1:].var().reset_index().rename(columns={'index': 'Column name', 0: 'Metrics'})
        true_var_avg_user_qual = var_avg_user_qual.loc[0]['Metrics']
        if self.sim_qual_cols:
          sim_var_avg_user_qual = var_avg_user_qual.loc[1:]['Metrics'].mean()
          table_row = pd.DataFrame({'Metrics': ['Variance of Average User BDs'], 'Simulations': [sim_var_avg_user_qual], 'MRT Data': [true_var_avg_user_qual]})
        else:
          table_row = pd.DataFrame({'Metrics': ['Variance of Average User BDs'], 'MRT Data': [true_var_avg_user_qual]})
        return table_row

    def get_avg_variances_within_user_brushing_duration(self):
        var_within_users = self.qualities.groupby('user_id').var().reset_index()
        avg_var_within_users = var_within_users.iloc[:, 1:].mean().reset_index().rename(columns={'index': 'Column name', 0: 'Metrics'})
        true_var_within_users = avg_var_within_users.loc[0]['Metrics']
        if self.sim_qual_cols:
          sim_var_within_users = avg_var_within_users.loc[1:]['Metrics'].mean()
          table_row = pd.DataFrame({'Metrics': ['Average of Variances of Within User BDs'], 'Simulations': [sim_var_within_users], 'MRT Data': [true_var_within_users]})
        else:
          table_row = pd.DataFrame({'Metrics': ['Average of Variances of Within User BDs'], 'MRT Data': [true_var_within_users]})
        return table_row

    def plot_nz_brushing_histogram(self):
        # Extract non-zero qualities for the true MRT data
        true_nz_qualities = self.qualities['quality'][self.qualities['quality'] != 0]

        # Extract non-zero qualities for the simulated data (if available)
        if self.sim_qual_cols:
            sim_nz_qualities = self.qualities[self.sim_qual_cols].apply(lambda x: x[x != 0].mean(), axis=1)
        else:
            sim_nz_qualities = pd.Series([])  # Empty Series if no simulated data

        # Calculate means
        true_mean = true_nz_qualities.mean()
        sim_mean = sim_nz_qualities.mean() if not sim_nz_qualities.empty else None

        # Create the panel plot with 2 subplots, sharing the y-axis
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # True MRT Data Histogram
        axes[0].hist(true_nz_qualities, bins=20, alpha=0.7, color='blue')
        axes[0].axvline(true_mean, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {true_mean:.2f}')
        axes[0].set_title('True MRT Data')
        axes[0].set_xlabel('Non-Zero Brushing Quality')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()

        # Simulated Data Histogram (if available)
        if not sim_nz_qualities.empty:
            axes[1].hist(sim_nz_qualities, bins=20, alpha=0.7, color='orange')
            axes[1].axvline(sim_mean, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {sim_mean:.2f}')
            axes[1].set_title('Simulated Data')
            axes[1].set_xlabel('Non-Zero Brushing Quality')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
        else:
            axes[1].set_title('Simulated Data')
            axes[1].set_xlabel('Non-Zero Brushing Quality')
            axes[1].set_ylabel('Frequency')
            axes[1].text(0.5, 0.5, 'No Simulated Data Available', horizontalalignment='center',
                         verticalalignment='center', transform=axes[1].transAxes, fontsize=12, color='gray')

        # Add a main title for the entire figure
        plt.suptitle('Histogram of Non-Zero Brushing Qualities', fontsize=16)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Show the plot
        plt.show()


    def make_table(self):
        table_row_1 = self.get_proportion_missed()
        table_row_2 = self.get_avg_nz_brushing_duration_pooled()
        table_row_3 = self.get_avg_nz_brushing_duration_unpooled()
        table_row_4 = self.get_variance_nz_brushing_duration_pooled()
        table_row_5 = self.get_variance_nz_brushing_duration_unpooled()
        table_row_6 = self.get_variance_avg_user_brushing_duration()
        table_row_7 = self.get_avg_variances_within_user_brushing_duration()
        return pd.concat([table_row_1, table_row_2, table_row_3, table_row_4, table_row_5, table_row_6, table_row_7], axis=0)

    def get_error_values(self):
      if not self.sim_qual_cols:
        return "This function requires simulation data to run."
      differences = self.qualities[self.sim_qual_cols].sub(self.qualities['quality'], axis=0)
      mse = (differences**2).mean(axis=0).mean()
      rmse = mse**0.5
      mae = (np.abs(differences)).mean(axis=0).mean()
      table_rows = pd.DataFrame({'Metrics': ['Mean Squared Error','Root Mean Squared Error','Mean Absolute Error'], 'Value': [mse,rmse,mae]})
      return table_rows

def get_ci(qualities, sim_qual_cols, confidence_level=0.95):
    qualities_copy = qualities.copy()

    means = qualities_copy[sim_qual_cols].mean(axis=1)
    stds = qualities_copy[sim_qual_cols].std(axis=1)

    z_score = np.abs(norm.ppf((1 - confidence_level) / 2))
    margin_of_error = z_score * stds
    lower_bounds = means - margin_of_error
    upper_bounds = means + margin_of_error

    within_interval = (qualities_copy['quality'] >= lower_bounds) & (qualities_copy['quality'] <= upper_bounds)
    qualities_copy.loc[:, 'within_interval'] = within_interval.astype(int)

    num_within_interval = within_interval.sum()

    #calculate the percentage of true values within the prediction interval
    percentage_within_interval = (num_within_interval / len(qualities_copy)) * 100

    return (f"Percentage of true values within the {confidence_level * 100}% prediction interval: {percentage_within_interval:.2f}%")

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

    user_ids = data['user_id'].str[6:9]
    true_means = data['quality']
    sim_means = data['sim_mean']
    lower_bounds = data['lower_bound']
    upper_bounds = data['upper_bound']

    ax.scatter(range(len(user_ids)), true_means, color='blue', label='True Mean')
    ax.scatter(range(len(user_ids)), sim_means, color='orange', label='Simulated Mean')

    for i in range(len(user_ids)):
        ax.errorbar(i, sim_means[i], yerr=[[sim_means[i] - lower_bounds[i]], [upper_bounds[i] - sim_means[i]]],
                    fmt='none', ecolor='black', capsize=5)

    ax.set_xlabel('User ID')
    ax.set_ylabel('Mean')
    ax.set_title(title)
    ax.set_xticks([])

    ax.legend()

true_nz_qualities = qualities['quality'][qualities['quality'] != 0]
sim_nz_qualities = qualities[sim_qual_cols].apply(lambda x: x[x != 0].mean(), axis=1)

avg_nz_qual = qualities.groupby('user_id').apply(lambda x: x.iloc[:, 1:][x.iloc[:, 1:] != 0].mean()).reset_index()
var_nz_qual = qualities.groupby('user_id').apply(lambda x: x.iloc[:, 1:][x.iloc[:, 1:] != 0].var()).reset_index()
prop_missed = qualities.groupby('user_id').apply(lambda x: (x == 0).mean()).drop(columns='user_id').reset_index()

fig, axs = plt.subplots(3, 1, figsize=(15, 15))

plot_ci_pooled_by_user(prop_missed, sim_qual_cols, axs[1], "Proportion of Missed Brushing Windows")
plot_ci_pooled_by_user(avg_nz_qual, sim_qual_cols, axs[0], "Average Nonzero Brushing Duration")
plot_ci_pooled_by_user(var_nz_qual, sim_qual_cols, axs[2], "Variance of Nonzero Brushing Duration")

plt.tight_layout()
plt.savefig('simulation_metrics_plot.png')  

sim_metrics_mrt = SimulationMetrics(qualities, sim_qual_cols)

result_table_mrt_1 = sim_metrics_mrt.make_table().to_latex(index=False)
with open('result_table_mrt_1.txt', 'w') as f:
    f.write(result_table_mrt_1)

result_table_mrt_2 = sim_metrics_mrt.get_error_values().to_latex(index=False)
with open('result_table_mrt_2.txt', 'w') as f:
    f.write(result_table_mrt_2)
