import numpy as np
import pandas as pd
# import scipy
from scipy.optimize import curve_fit
from scipy.stats import linregress

from sklearn.linear_model import LinearRegression

def read_json(filename):
    #read json into panda df
    rawdata_pd = pd.read_json(filename)
    return  rawdata_pd

def check_tau_consistency(data_pd):
    lefthem_idx = data_pd.loc[(data_pd['hemisphere'] == 'left ACx') & ~data_pd['logtau_est'].isnull()]
    righthem_idx = data_pd.loc[(data_pd['hemisphere'] == 'right ACx') & ~data_pd['logtau_est'].isnull()]
    # print("left hemi idx -- ", lefthem_idx)

    lefthem_taus = lefthem_idx['tau_est'].values
    righthem_taus = righthem_idx['tau_est'].values

    # for each hemisphere's data -- get the median of Tau
    lefthem_taus_median = np.median(lefthem_taus)
    righthem_taus_median = np.median(righthem_taus)

    # print("left hemi taus: ", lefthem_taus)
    # print("right hemi taus: ", righthem_taus)
    print(f'left hemi tau median: {lefthem_taus_median}; right hemi tau median:\
        {righthem_taus_median}')

    # if output is too weird then prune out all the bad estimates of tau -- >300ms and <20ms and
    # then do average
    lefthem_idx_pruned = lefthem_idx.loc[(lefthem_idx['tau_est']<=300.0) &
            (lefthem_idx['tau_est']>=10)]
    righthem_idx_pruned = righthem_idx.loc[(righthem_idx['tau_est']<=300.0) &
            (righthem_idx['tau_est']>=10)]
    lefthem_taus_pruned = lefthem_idx_pruned['tau_est'].values
    righthem_taus_pruned = righthem_idx_pruned['tau_est'].values

    lefthem_tau_prunedmean = np.mean(lefthem_taus_pruned)
    righthem_tau_prunedmean = np.mean(righthem_taus_pruned)

    print(f'left hemi pruned mean: {lefthem_tau_prunedmean}; right hemi pruned mean:\
        {righthem_tau_prunedmean}')

def linear_func(x, m, c):
    return m*x + c

def check_fr_tau_corr(data_pd):
    # weighted regression between fr and tc 
    data_pd_notnan = data_pd.loc[~data_pd['logtau_est'].isnull()]
    firing_rates = data_pd_notnan['firing_rate'].values
    taus = data_pd_notnan['tau_corrected'].values
    # covarinance between fr and tau
    simplecov = np.cov(firing_rates, taus)
    print(f'covariance of fr and tau {simplecov}')

    sigmas = data_pd_notnan['var_logtau'].values
    # print("firing rates -- ", firing_rates)
    # print("taus -- ", taus)
    p0 = [1.0, -1.0]
    corr_prm, pcov = curve_fit(linear_func, firing_rates, taus, p0, sigmas)
    print("linear curve fit - slope with sigmas: ", corr_prm)
    # print("cov matrix: ", pcov)

    # weights = np.reciprocal(data_pd_notnan['var_logtau'].values)
    # weights_norm = weights/np.sum(weights)

    # scikit lin regressioin
    # reg = LinearRegression().fit(firing_rates, taus)
    # r2score = reg.score(firing_rates, taus)
    # print("r2 score: ", r2score)

    #scipy linreg
    res = linregress(firing_rates, taus)
    print("slope: ", res.slope)
    print(f"R-squared: {res.rvalue**2:.6f}")


if (__name__ == "__main__"):
    jsonfile = "../data/data_tau_bias_variance.json"
    data_pd = read_json(jsonfile)
    # print(data_pd)
    # print(list(data_pd.columns.values))
    # ['dtbin', 'set', 'hemisphere', 'n_spikes', 'n_trials', 'mean_spikes', 'sd_spikes',
    # 'Abin_est', 'tau_est', 'mse_lsq', 'logtau_est', 'autocorr', 'mse_mean', 'r2',
    # 'bias_logtau', 'var_logtau', 'logtau_corrected', 'tau_corrected', 'duration', 'firing_rate']


    check_tau_consistency(data_pd)
    check_fr_tau_corr(data_pd)
