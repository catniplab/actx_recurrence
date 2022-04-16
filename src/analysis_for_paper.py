import numpy as np
import pandas as pd
# import scipy
from scipy.optimize import curve_fit
from scipy.stats import linregress

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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
    lefthem_tau_prunedstd = np.std(lefthem_taus_pruned)
    righthem_tau_prunedstd = np.std(righthem_taus_pruned)

    print(f'left hemi pruned mean: {lefthem_tau_prunedmean}, std {lefthem_tau_prunedstd}; right hemi pruned mean:\
            {righthem_tau_prunedmean}, std: {righthem_tau_prunedstd}')

def linear_func(x, m, c):
    return m*x + c

def check_fr_tau_corr(data_pd):
    # weighted regression between fr and tc 
    data_pd_notnan = data_pd.loc[~data_pd['logtau_est'].isnull()]
    firing_rates = data_pd_notnan['firing_rate'].values
    taus = data_pd_notnan['tau_corrected'].values
    # print("firing rates -- ", firing_rates)
    # print("taus -- ", taus)
    # print("taus min and max -- ", np.min(taus), np.max(taus))
    # print("sorted taus -- ", np.sort(taus))

    # covarinance between fr and tau
    simplecov = np.cov(firing_rates, taus)
    # print(f'covariance of fr and tau {simplecov}')

    # scipy curve fitting
    sigmas = data_pd_notnan['var_logtau'].values
    p0 = [1.0, -1.0]
    corr_prm, pcov = curve_fit(linear_func, firing_rates, taus, p0, sigmas)
    print("linear curve fit - slope with sigmas: ", corr_prm)
    print("cov matrix after curve fitting : ", pcov)
    taus_fitted = linear_func(firing_rates, corr_prm[0], corr_prm[1])
    # cov_coeff = 1 - np.sum((taus - taus_fitted)**2)/np.sum((taus - np.mean(taus))**2)
    cov_coeff = np.sum((taus_fitted - np.mean(taus))**2)/np.sum((taus - np.mean(taus))**2)
    print("R2 value: ", cov_coeff)

    # print("r2 check shapes: ", taus.shape, taus_fitted.shape)
    # print("r2 check tau mean -- ", np.mean(taus))
    # print("r2 check - y hat -- ", taus_fitted)
    # print("r2 value check -- ", (taus - taus_fitted)**2, np.sum((taus - taus_fitted)**2))
    # print("r2 value check 2 -- ", (taus - np.mean(taus))**2, np.sum((taus - np.mean(taus))**2))

    # scikit lin regressioin
    # weights = np.reciprocal(data_pd_notnan['var_logtau'].values)
    # weights_norm = weights/np.sum(weights)
    # reg = LinearRegression().fit(firing_rates, taus)
    # r2score = reg.score(firing_rates, taus)
    # print("r2 score: ", r2score)

    #scipy linreg
    res = linregress(firing_rates, taus)
    print("slope: ", res.slope)
    print(f"R-squared: {res.rvalue**2:.6f}")

    # plotting the fr vs tau linear regression line
    plot_raster(firing_rates, taus, corr_prm[0], corr_prm[1], '../outputs/analysis_frvstau.pdf')
    plot_raster(firing_rates, taus, res.slope, res.intercept,\
            '../outputs/analysis_frvstau_nonweighted.pdf')

    # check the tau estimate from fr mean of each hemisphere
    lefthem_idx = data_pd.loc[(data_pd['hemisphere'] == 'left ACx') & ~data_pd['logtau_est'].isnull()]
    righthem_idx = data_pd.loc[(data_pd['hemisphere'] == 'right ACx') & ~data_pd['logtau_est'].isnull()]

    lefthem_taus = np.mean(lefthem_idx['tau_corrected'].values)
    righthem_taus = np.mean(righthem_idx['tau_corrected'].values)
    lefthem_fr = np.mean(lefthem_idx['firing_rate'].values)
    righthem_fr = np.mean(righthem_idx['firing_rate'].values)

    lefthem_tau_fitted = linear_func(lefthem_fr, corr_prm[0], corr_prm[1])
    righthem_tau_fitted = linear_func(righthem_fr, corr_prm[0], corr_prm[1])
    
    print(f'left hemi -- tau corrected: {lefthem_taus}; tau fitted: {lefthem_tau_fitted}')
    print(f'right hemi -- tau corrected: {righthem_taus}; tau fitted: {righthem_tau_fitted}')

    # checking with pruned inferences
    lefthem_idx_pruned = lefthem_idx.loc[(lefthem_idx['tau_corrected']<=300.0) &
            (lefthem_idx['tau_corrected']>=10)]
    righthem_idx_pruned = righthem_idx.loc[(righthem_idx['tau_corrected']<=300.0) &
            (righthem_idx['tau_corrected']>=10)]

    lefthem_taus_pruned = lefthem_idx_pruned['tau_corrected'].values
    righthem_taus_pruned = righthem_idx_pruned['tau_corrected'].values
    lefthem_fr_pruned = lefthem_idx_pruned['firing_rate'].values
    righthem_fr_pruned = righthem_idx_pruned['firing_rate'].values

    lefthem_tau_prunedmean = np.mean(lefthem_taus_pruned)
    righthem_tau_prunedmean = np.mean(righthem_taus_pruned)
    lefthem_tau_prunedstd = np.std(lefthem_taus_pruned)
    righthem_tau_prunedstd = np.std(righthem_taus_pruned)

    lefthem_tau_fitted = linear_func(lefthem_fr, corr_prm[0], corr_prm[1])
    righthem_tau_fitted = linear_func(righthem_fr, corr_prm[0], corr_prm[1])
    print(f'for pruned combined data: left hemi -- inferred mean: {lefthem_tau_prunedmean}; fitted:\
        {lefthem_tau_fitted}')
    print(f'for pruned combined data: right hemi -- inferred mean: {righthem_tau_prunedmean}; fitted:\
        {righthem_tau_fitted}')

def plot_raster(firing_rates, taus, m, c, filename):
    plt.scatter(firing_rates, taus)

    X = np.linspace(np.min(firing_rates), max(firing_rates), 100)
    Y = linear_func(X, m, c)
    plt.plot(X, Y, 'r-')
    plt.xlabel('firing rates')
    plt.ylabel('taus')
    plt.savefig(filename)
    plt.close()
    # plt.plot()
    # plt.show()

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
