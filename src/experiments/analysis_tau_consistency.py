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
    print(f'left hemi sample count: {lefthem_idx.shape[0]}; right hemi sample count:\
        {righthem_idx.shape[0]}')
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

    lefthem_tau_pruned_stderror = lefthem_tau_prunedstd/np.sqrt(lefthem_idx.shape[0])
    righthem_tau_pruned_stderror = righthem_tau_prunedstd/np.sqrt(righthem_idx.shape[0])

    print(f'left hemi -- pruned mean: {lefthem_tau_prunedmean}, std error of mean:\
                {lefthem_tau_pruned_stderror}, n = {lefthem_idx_pruned.shape[0]}')
    print(f'Right hemi -- pruned mean: {righthem_tau_prunedmean}, std err of mean:\
            {righthem_tau_pruned_stderror}, n = {righthem_idx_pruned.shape[0]}')

def find_var_from_logvar(logvar, data):
    # form found using the moment generators
    mux = data['logtau_corrected'].values + data['bias_logtau'].values
    # print("mux:", mux)
    varx = logvar
    # return np.exp(2*mux+2*varx) - np.exp(2*mux+varx)
    return np.exp(2*mux+varx) * (np.exp(varx) - 1)

def linear_func(x, m, c):
    return m*x + c

def check_fr_tau_corr(data_pd):

    # weighted regression between fr and log tc 
    data_pd_notnan = data_pd.loc[~data_pd['logtau_est'].isnull()]
    firing_rates = data_pd_notnan['firing_rate'].values
    logtaus = data_pd_notnan['logtau_corrected'].values
    # print("firing rates -- ", firing_rates)
    # print("taus -- ", taus)
    # print("taus min and max -- ", np.min(taus), np.max(taus))
    # print("sorted taus -- ", np.sort(taus))

    # covarinance between fr and tau
    # simplecov = np.cov(firing_rates, logtaus)
    # print(f'covariance of fr and tau {simplecov}')

    # scipy curve fitting
    # sigmas = (data_pd_notnan['var_logtau'].values)**(0.5)
    # p0 = [1.0, -1.0]
    # corr_prm, pcov = curve_fit(linear_func, firing_rates, logtaus, p0, sigmas)
    # logtaus_fitted = linear_func(firing_rates, corr_prm[0], corr_prm[1])
    # # cov_coeff = 1 - np.sum((taus - taus_fitted)**2)/np.sum((taus - np.mean(taus))**2)
    # cov_coeff = np.sum((logtaus_fitted - np.mean(logtaus))**2)/np.sum((logtaus - np.mean(logtaus))**2)
    # print(f' scipy curve fit -- slope: {corr_prm[0]}, intercept: {corr_prm[1]}')
    # print("R2 value: ", cov_coeff)


    # scikit lin regression
    weights = np.reciprocal(data_pd_notnan['var_logtau'].values)
    res = LinearRegression().fit(firing_rates.reshape(-1,1), logtaus.reshape(-1,1), weights)
    r2score = res.score(firing_rates.reshape(-1,1), logtaus.reshape(-1,1), weights)
    print(f' scikit: FR vs Log Taus -- slope: {res.coef_[0][0]}, intercept: {res.intercept_}')
    print("r2 score from scikit learn: ", r2score)

    # linear regression between fr and time constant (not log)
    taus = data_pd_notnan['tau_corrected'].values
    sigma_taus = find_var_from_logvar(data_pd_notnan['var_logtau'].values, data_pd_notnan)**0.5
    # print('sigma taus: ', sigma_taus)
    data_pd_notnan['var_tau'] = sigma_taus**2
    # data_pd_notnan.insert(16, 'var_tau', sigma_taus)
    weights_taus = np.reciprocal(sigma_taus)
    # weights_taus = np.reciprocal(data_pd_notnan['var_logtau'].values)
    res_taus = LinearRegression().fit(firing_rates.reshape(-1,1), taus.reshape(-1,1), weights_taus)
    r2score_taus = res_taus.score(firing_rates.reshape(-1,1), taus.reshape(-1,1), weights_taus)
    print(f' scikit: FR vs Taus -- slope: {res_taus.coef_[0][0]}, intercept: {res_taus.intercept_}')
    print("r2 score from scikit learn: ", r2score_taus)

    # slope and intercept on logtau analysis
    m_logtau, c_logtau = res.coef_[0][0], res.intercept_[0]

    # check the tau estimate from fr mean of each hemisphere
    lefthem_idx = data_pd.loc[(data_pd['hemisphere'] == 'left ACx') & ~data_pd['logtau_est'].isnull()]
    righthem_idx = data_pd.loc[(data_pd['hemisphere'] == 'right ACx') & ~data_pd['logtau_est'].isnull()]

    lefthem_logtaus = np.mean(lefthem_idx['logtau_corrected'].values)
    righthem_logtaus = np.mean(righthem_idx['logtau_corrected'].values)
    lefthem_fr = np.mean(lefthem_idx['firing_rate'].values)
    righthem_fr = np.mean(righthem_idx['firing_rate'].values)

    lefthem_logtau_fitted = linear_func(lefthem_fr, m_logtau, c_logtau)
    righthem_logtau_fitted = linear_func(righthem_fr, m_logtau, c_logtau)
    
    print("for all corrected estimations without pruning")
    print(f'left hemi -- logtau corrected: {lefthem_logtaus}; logtau fitted: {lefthem_logtau_fitted}')
    print(f'right hemi -- logtau corrected: {righthem_logtaus}; logtau fitted: {righthem_logtau_fitted}')

    # checking with pruned inferences
    lefthem_idx_pruned = lefthem_idx.loc[(lefthem_idx['tau_corrected']<=300.0) &
            (lefthem_idx['tau_corrected']>=10)]
    righthem_idx_pruned = righthem_idx.loc[(righthem_idx['tau_corrected']<=300.0) &
            (righthem_idx['tau_corrected']>=10)]

    lefthem_logtaus_pruned = lefthem_idx_pruned['logtau_corrected'].values
    righthem_logtaus_pruned = righthem_idx_pruned['logtau_corrected'].values
    lefthem_fr_pruned = np.mean(lefthem_idx_pruned['firing_rate'].values)
    righthem_fr_pruned = np.mean(righthem_idx_pruned['firing_rate'].values)

    lefthem_logtau_prunedmean = np.mean(lefthem_logtaus_pruned)
    righthem_logtau_prunedmean = np.mean(righthem_logtaus_pruned)
    lefthem_logtau_prunedstd = np.std(lefthem_logtaus_pruned)
    righthem_logtau_prunedstd = np.std(righthem_logtaus_pruned)

    lefthem_logtau_fitted = linear_func(lefthem_fr_pruned, m_logtau, c_logtau)
    righthem_logtau_fitted = linear_func(righthem_fr_pruned, m_logtau, c_logtau)

    # TODO: should these means be wighted as well??
    mean_plots = {
            'left_mean_fr':lefthem_fr_pruned,
            'left_fit_logtau':lefthem_logtau_fitted,
            'left_mean_logtau': lefthem_logtaus,
            'right_mean_fr':righthem_fr_pruned,
            'right_fit_logtau':righthem_logtau_fitted,
            'right_mean_logtau': righthem_logtaus
        }

    print("for pruned corrected data: ")
    print(f'left hemi -- inferred mean: {lefthem_logtau_prunedmean}; fitted:\
        {lefthem_logtau_fitted}')
    print(f'right hemi -- inferred mean: {righthem_logtau_prunedmean}; fitted:\
        {righthem_logtau_fitted}')

    left_hem_tau_pred = np.mean(res_taus.predict(lefthem_idx['firing_rate'].values.reshape(-1,1)))
    right_hem_tau_pred = np.mean(res_taus.predict(righthem_idx['firing_rate'].values.reshape(-1,1)))
    pred_diff = np.abs(left_hem_tau_pred-right_hem_tau_pred)
    print(f'left hem mean tau predicted from fr = {left_hem_tau_pred},\
            right hem mean tau predicted from fr = {right_hem_tau_pred}')
    print(f'predicted difference : {pred_diff}')

    # plotting the fr vs tau linear regression line
    # plot_frvstau(firing_rates, logtaus, corr_prm[0], corr_prm[1],
            # '../outputs/analysis_frvstau_fromcurvefit.pdf')
    # --------- Plotting ------------
    plot_frvstau_fancy(data_pd_notnan, firing_rates, logtaus, res, res_taus, mean_plots,
            '../../outputs/analysis_frvstau_fancy.pdf')

def plot_frvstau(firing_rates, taus, m, c, filename):
    plt.scatter(firing_rates, taus)
    X = np.linspace(np.min(firing_rates), max(firing_rates), 100)
    Y = linear_func(X, m, c)
    plt.plot(X, Y, 'r-')
    plt.xlabel('firing rates')
    plt.ylabel('log taus')
    plt.savefig(filename)
    # plt.plot()
    # plt.show()
    plt.close()

def plot_frvstau_fancy(data, firing_rates, taus, res, res_tau, mean_plots, filename):

    # plot left and right taus with error
    m_logtau, c_logtau = res.coef_[0][0], res.intercept_[0]
    m_tau, c_tau = res_tau.coef_[0][0], res_tau.intercept_[0]

    lefthem_idx = data.loc[(data['hemisphere'] == 'left ACx') & ~data['logtau_est'].isnull()]
    righthem_idx = data.loc[(data['hemisphere'] == 'right ACx') & ~data['logtau_est'].isnull()]

    # plt.scatter(lefthem_idx['firing_rates'], lefthem_idx['logtau_corrected'], color='#ff6600',
            # alpha=0.7)
    leftplot_errlims_y = [lefthem_idx['tau_corrected'] - np.exp(lefthem_idx['logtau_corrected'] -\
            (lefthem_idx['var_logtau'])**0.5), np.exp(lefthem_idx['logtau_corrected'] +\
            (lefthem_idx['var_logtau'])**0.5) - lefthem_idx['tau_corrected']] 

    rightplot_errlims_y = [righthem_idx['tau_corrected'] - np.exp(righthem_idx['logtau_corrected'] -\
            (righthem_idx['var_logtau'])**0.5), np.exp(righthem_idx['logtau_corrected'] +\
            (righthem_idx['var_logtau'])**0.5) - righthem_idx['tau_corrected']] 

    # left_hem_stderr = lefthem_idx['sd_spikes']/((lefthem_idx['duration']/\
                    # (lefthem_idx['dtbin']**0.5))*(lefthem_idx['n_trials']**0.5))
    left_hem_stderr = lefthem_idx['sd_spikes']/(lefthem_idx['n_trials']**0.5)
    # print("max err in left: ", lefthem_idx['sd_spikes']/(lefthem_idx['n_trials']**0.5),\
            # lefthem_idx['sd_spikes'], lefthem_idx['n_trials'])
    # leftplot_errlims_x = np.array([lefthem_idx['firing_rate'].values - left_hem_stderr.values,\
            # lefthem_idx['firing_rate'].values + left_hem_stderr.values])
    leftplot_errlims_x = np.array(left_hem_stderr.values)
    # print(leftplot_errlims_x)
    # leftplot_errlims_x = [lefthem_idx['firing_rate'] - lefthem_idx['sd_spikes']\
            # /lefthem_idx['duration'], lefthem_idx['sd_spikes']/lefthem_idx['duration']\
            # - lefthem_idx['firing_rate']] 

    # right_hem_stderr = righthem_idx['sd_spikes']/((righthem_idx['duration']/\
                    # (righthem_idx['dtbin']**0.5))*(righthem_idx['n_trials']**0.5))
    right_hem_stderr = righthem_idx['sd_spikes']/(righthem_idx['n_trials']**0.5)
    # print("max err in right: ", righthem_idx['sd_spikes']/(righthem_idx['n_trials']**0.5),\
            # righthem_idx['sd_spikes'], righthem_idx['n_trials'])
    # rightplot_errlims_x = np.array([righthem_idx['firing_rate'].values - right_hem_stderr.values,\
            # righthem_idx['firing_rate'].values + right_hem_stderr.values])
    rightplot_errlims_x = np.array(right_hem_stderr.values)
    # print("right hemi fr std -- ", righthem_idx['sd_spikes']/((righthem_idx['duration']/\
                    # (righthem_idx['dtbin']**0.5))*(righthem_idx['n_trials']**0.5)) )

    allerrs = np.concatenate((leftplot_errlims_x, rightplot_errlims_x), axis=0)
    # print(allerrs)
    maxerr = np.max(allerrs)
    print("max err = ", maxerr, "spks")
    print("max err = ", maxerr/(lefthem_idx['duration'][0]/1000), "spks/s")

    plt.errorbar(lefthem_idx['firing_rate'], np.exp(lefthem_idx['logtau_corrected']), #lefthem_idx['tau_corrected'],
                yerr=leftplot_errlims_y, xerr=leftplot_errlims_x, fmt='o', color='#4f94c4',\
                elinewidth=1, capsize=5, alpha=0.7, label='left hem')
    # plt.scatter(mean_plots['left_mean_fr'], np.exp(mean_plots['left_fit_logtau']), marker='*',\
            # color='#4f94c4', alpha=0.95, label = 'left hem mean prdicted')
    plt.scatter(mean_plots['left_mean_fr'], np.exp(mean_plots['left_mean_logtau']), marker='*',\
            color='#4f94c4', alpha=0.95, label = 'left hem mean fr,tau')

    plt.errorbar(righthem_idx['firing_rate'], righthem_idx['tau_corrected'],
            yerr=rightplot_errlims_y, xerr=rightplot_errlims_x, fmt='o', color='#ff851a',\
            elinewidth=1, capsize=5, alpha=0.7, label='right hem')
    # plt.scatter(mean_plots['right_mean_fr'], np.exp(mean_plots['right_fit_logtau']), marker='*',\
            # color='#ff851a', alpha=0.95, label='right hem mean predicted')
    plt.scatter(mean_plots['right_mean_fr'], np.exp(mean_plots['right_mean_logtau']), marker='*',\
            color='#ff851a', alpha=0.95, label = 'right hem mean fr,tau')

    # plt.errorbar(lefthem_idx['firing_rate'], lefthem_idx['tau_corrected'],
        # yerr=lefthem_idx['var_logtau'], fmt='o', color='#4f94c4', elinewidth=1, capsize=5, alpha=0.7)
    # plt.errorbar(righthem_idx['firing_rate'], righthem_idx['tau_corrected'],
        # yerr=righthem_idx['var_logtau'], fmt='o', color='#ff851a', elinewidth=1, capsize=5,
        # alpha=0.7)

    # plt.scatter(firing_rates, taus)
    X = np.linspace(np.min(firing_rates), max(firing_rates), 100)
    Y_logtau = np.exp(linear_func(X, m_logtau, c_logtau))
    Y_tau = linear_func(X, m_tau, c_tau)
    plt.plot(X, Y_tau, '-', color='#949494', alpha=0.6, linewidth=2, label='X vs Tau fit')
    plt.plot(X, Y_logtau, '--', color='#616161', alpha=0.6, linewidth=2, label='X vs Log Tau fit')
    plt.xlabel('firing rates')
    plt.ylabel('taus')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='upper right', fontsize=4, markerscale=0.5)
    plt.savefig(filename)
    # plt.plot()
    # plt.show()
    plt.close()

if (__name__ == "__main__"):
    jsonfile = "../../data/data_tau_bias_variance.json"
    data_pd = read_json(jsonfile)
    # print(data_pd)
    # print(list(data_pd.columns.values))
    # ['dtbin', 'set', 'hemisphere', 'n_spikes', 'n_trials', 'mean_spikes', 'sd_spikes',
    # 'Abin_est', 'tau_est', 'mse_lsq', 'logtau_est', 'autocorr', 'mse_mean', 'r2',
    # 'bias_logtau', 'var_logtau', 'logtau_corrected', 'tau_corrected', 'duration', 'firing_rate']

    print("---- checking tau consistency ----")
    check_tau_consistency(data_pd)

    print("---- FR anf Tau correlation ----")
    check_fr_tau_corr(data_pd)
