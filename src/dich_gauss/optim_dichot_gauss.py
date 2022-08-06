import numpy as np
from scipy.stats import norm, multivariate_normal as mnorm
from scipy.special import erfinv, erf
import scipy.optimize
import scipy.linalg
import matplotlib.pyplot as plt
# from IPython.display import clear_output

import warnings


class WarningDGOpt(UserWarning):
    pass



def get_bivargauss_cdf(vals, corr_coef):
    """
    Computes cdf of a bivariate Gaussian distribution with mean zero, variance 1 and input correlation.

    Inputs:
        :param vals: arguments for bivariate cdf (μi, μj).
        :param corr_coef: correlation coefficient of biavariate Gaussian (Λij).

    Returns:
        :return: Φ2([μi, μj], Λij)
    """
    cov = np.eye(2)
    cov[1, 0], cov[0, 1] = corr_coef, corr_coef
    cdf = mnorm.cdf(vals, mean=[0., 0.], cov=cov)
    # print("cdf val: ", cdf, corr_coef)
    return cdf


# def function(gauss_covar, data_means, gauss_means, data_covar):
def function(data_means, gauss_means, data_covar, gauss_covar):
    """
    Computes the pairwise covariance eqn for root finding algorithm.

    Inputs:
        :param data_means: mean of binary spike train of 2 neurons (ri, rj).
        :param gauss_means: mean of bivariate Gaussian that calculated from data for the 2 neurons (μi, μj).
        :param data_covar: covariance between the spike trains of the 2 neurons (Σij).
        :param gauss_covar: covariance of the bivariate Gaussian distribution corresponding to the 2 neurons (Λij).

    Returns:
        :return: Φ2([μi, μi], Λij) - ri*rj - Σij
    """
    bivar_gauss_cdf = np.mean(get_bivargauss_cdf(vals=np.array(gauss_means).T,
                                                 corr_coef=gauss_covar))

    # print("cdf calculations: ", bivar_gauss_cdf, np.prod(data_means), data_covar)
    return bivar_gauss_cdf - np.prod(data_means) - data_covar

    # bivar_gauss_cdf = np.mean(get_bivargauss_cdf(vals=np.array(data_means).T,
                                                 # corr_coef=gauss_covar))
    # print("cdf calculations: ", bivar_gauss_cdf,  data_covar)
    # return data_covar - bivar_gauss_cdf 

# def function(gauss_covar, data_means, gauss_means, data_covar):
def gauss_cov_func(data_means, gauss_means, data_covar, gauss_covar):
    """
    Computes the pairwise covariance eqn for root finding algorithm.

    Inputs:
        :param data_means: mean of binary spike train of 2 neurons (ri, rj).
        :param gauss_means: mean of bivariate Gaussian that calculated from data for the 2 neurons (μi, μj).
        :param data_covar: covariance between the spike trains of the 2 neurons (Σij).
        :param gauss_covar: covariance of the bivariate Gaussian distribution corresponding to the 2 neurons (Λij).

    Returns:
        :return: Φ2([μi, μi], Λij) - ri*rj - Σij
    """
    gauss_cov = np.eye(2)
    gauss_cov[1, 0], gauss_cov[0, 1] = gauss_covar, gauss_covar
    cdf = mnorm.cdf(gauss_means, mean=[0., 0.], cov=gauss_cov)
    # cdf = mnorm.cdf([0., 0.], mean=gauss_means, cov=gauss_cov)
    # print("cdf val: ", cdf, corr_coef)

    bivar_gauss_cdf = np.mean(cdf)

    # print("cdf calculations: ", bivar_gauss_cdf, np.prod(data_means), data_covar)
    return bivar_gauss_cdf - np.prod(data_means) - data_covar

    # bivar_gauss_cdf = np.mean(get_bivargauss_cdf(vals=np.array(data_means).T,
                                                 # corr_coef=gauss_covar))
    # print("cdf calculations: ", bivar_gauss_cdf,  data_covar)
    # return data_covar - bivar_gauss_cdf 

def find_root_bisection(*eqn_input, eqn=function, maxiters=1000, tol=1e-10):
    """
    Finds root of input equation using the bisection algorithm.

    Inputs:
        :param eqn_input: list containing inputs to \'eqn\' method.
        :param eqn: method implementing the equation for which we need the root.
        :param maxiters: max. number of iterations for bisection algorithm.
        :param tol: tolerance value for convergence of bisection algorithm.

    Returns:
        :return: root of \'eqn\'.
    """
    λ0 = -.999999999
    λ1 = .999999999

    f0 = eqn(*eqn_input, λ0)
    f1 = eqn(*eqn_input, λ1)

    # print('f0, f1', f0, f1)

    if np.abs(f0) < tol:
        warnings.warn("Warning: f0 is already close to 0. Returning initial value.", WarningDGOpt)
        return λ0

    if np.abs(f1) < tol:
        warnings.warn("Warning: f1 is already close to 0. Returning initial value.", WarningDGOpt)
        return λ1

    if f0 * f1 > tol:
        warnings.warn('Warning: Both initial covariance values lie on same side of zero crossing. '
                      'Setting value to 0.',
                      WarningDGOpt)
        # print('f0, f1', f0, f1)
        λ = 0.
        return λ

    f = np.inf
    it = 0
    while np.abs(f) > tol and it < maxiters:
        λ = (λ0 + λ1) / 2
        f = eqn(*eqn_input, λ)

        # print('λ, f(λ)', λ, f)

        if f > 0:
            λ1 = λ
        elif f < 0:
            λ0 = λ
        it += 1
    # clear_output(wait=True)
    return λ

def plot_func_outputs(eqn=function):
    plot_path = "../dichgauss_func_trace.pdf"
    mfr = 0.5
    dg_mfr = norm.ppf(mfr)
    dg_cov_0 = -0.9999
    dg_cov_1 = 0.9999

    cov_range = np.arange(-0.9999, 0.9999, 0.01)
    
    cov_0_arr = []
    cov_1_arr = []
    for i in range(cov_range.shape[0]):
        cov_0_arr.append(eqn([mfr]*2, [dg_mfr]*2, cov_range[i], dg_cov_0))
        cov_1_arr.append(eqn([mfr]*2, [dg_mfr]*2, cov_range[i], dg_cov_1))

    fig, ax = plt.subplots()
    ax.plot(cov_range, cov_0_arr, 'r', label='-0.99999')
    ax.plot(cov_range, cov_1_arr, 'b', label='0.99999')
    ax.plot(cov_range, np.array(cov_0_arr)*np.array(cov_1_arr), 'g', label='product')
    ax.set_xlabel('range of cov values')
    ax.set_ylabel('range of function values')
    ax.legend(fancybox=True)
    # plt.show()
    plt.savefig(plot_path)


def plot_dg_outputs(eqn=function):
    plot_path = "../phi2_func_trace.pdf"
    mfr = [0.75, 0.5, 0.25, 0.1, 0.01, 0.001]
    colors = ['r', 'b', 'g', 'c', 'm', 'y']
    # dg_mfr = norm.ppf(mfr)
    # cov = 0.2
    pedastal = 1e-5
    tau = 0.1
    binsize = 0.02 #s
    time_i = 1
    autocov0 = (np.array(mfr)-pedastal)*np.exp(-1*(time_i*binsize)/tau) + pedastal
    # dg_cov_0 = -0.9999
    # dg_cov_1 = 0.9999

    dg_cov_range = np.linspace(-0.999999, 0.999999, 1000)
    dg_cov_range[-1] = 0.999999999
    dg_cov_range[0] = -0.999999999
    
    phi2s_arr = []
    f_0s_arr = []
    dg_mfrs=[]
    for j in range(len(mfr)):
        dg_mfr = norm.ppf(mfr[j])
        dg_mfrs.append(dg_mfr)
        phi2_arr = []
        f_0_arr = []
        for i in range(dg_cov_range.shape[0]):
            # f_0_arr.append(eqn([mfr[j]]*2, [dg_mfr]*2, cov, dg_cov_range[i])-mfr[j])
            phi2_arr.append(get_bivargauss_cdf([dg_mfr]*2, dg_cov_range[i]))
            # f_0_arr.append(get_bivargauss_cdf([dg_mfr]*2, dg_cov_range[i])-mfr[j]**2)
            f_0_arr.append(get_bivargauss_cdf([dg_mfr]*2, dg_cov_range[i])-mfr[j]**2-autocov0[j])
        phi2s_arr.append(phi2_arr)
        f_0s_arr.append(f_0_arr)

    # print(f_0s_arr[0])

    fig, ax = plt.subplots()
    for i in range(len(mfr)):
        ax.plot(dg_cov_range, phi2s_arr[i], color=colors[i],\
                label='\Phi2 for mfr={}, dg_mean={:.3e}'.format(mfr[i],dg_mfrs[i]))
        ax.plot(dg_cov_range, f_0s_arr[i], color=colors[i],\
                label='\Si2-\lam for mfr={}, lam={:.3e}'.format(mfr[i], autocov0[i]), linestyle='--')
    # ax.plot(dg_cov_range, f_0_arr, 'r', label='\Si - cov evaluation')
        # f_1_arr = []
    # ax.plot(cov_range, np.array(cov_0_arr)*np.array(cov_1_arr), 'g', label='product')
    ax.set_xlabel('range of covar values')
    ax.set_ylabel('range of bivar cdf values')
    ax.legend(fancybox=True, fontsize='x-small')
    plt.show()
    # plt.title('covariance checked for: {}'.format(autocov0))
    # plt.savefig(plot_path)

def test_dich_gauss():
    lam1 = 0.5
    lam2 = 0.5

    cov = 0.125
    # cov = cov/2

    gamma1 = norm.ppf(lam1)
    gamma2 = norm.ppf(lam2)

    cov_dg = find_root_bisection([lam1, lam2], [gamma1, gamma2], cov, eqn=gauss_cov_func)
    # cov_dg = find_gauss_covar([lam1, lam2], [gamma1, gamma2], cov)

    print(f'gamma1: {gamma1}, gamma2: {gamma2}, cov_dg: {cov_dg}')

    # test_output = function([lam1, lam2], [gamma1, gamma2], 0.0, 0.39)
    test_output = gauss_cov_func([lam1, lam2], [gamma1, gamma2], 0.0, 0.707)
    print(test_output)

class DGOptimise(object):
    """
        Finds the parameters of the multivariate Gaussian that best fit the given binary spike train.
        Inputs:
            :param data: binary spike count data of size timebins x repeats x neurons
    """

    def __init__(self, data):
        self.timebins, self.trials, self.num_neur = data.shape
        self.tril_inds = np.tril_indices(self.num_neur, -1)
        self.data = data

    @property
    def gauss_mean(self):
        """
        Computes mean of the multivariate Gaussian corresponding to the input binary spike train.
        """
        data = self.data

        mean = data.mean(1)
        self._check_mean(mean)  # Check if mean lies between 0 and 1

        # Need this to ensure inverse cdf calculation (norm.ppf()) does not break
        mean[mean == 0.] += 1e-4
        mean[mean == 1.] -= 1e-4

        gauss_mean = norm.ppf(mean)
        return gauss_mean

    @property
    def data_tvar_covariance(self):
        """Computes covariance between spike trains from different neurons, averaged across timebins and trials.
           Calculated for time-varying firing rate"""
        data = self.data

        data_norm = (data - data.mean(0)).reshape(self.timebins, -1)
        tot_covar = data_norm.T.dot(data_norm).reshape(self.trials, self.num_neur, self.trials, self.num_neur)
        inds = range(self.trials)
        tot_covar = tot_covar[inds, :, inds, :].mean(0) / self.timebins
        return tot_covar

    @property
    def data_tfix_covariance(self):
        """Computes covariance between spike trains from different neurons, averaged across repeats. Calculated for
           fixed firing rate."""
        data = self.data
        data_norm = (data - data.mean(1)).reshape(-1, self.num_neur)
        tot_covar = data_norm.T.dot(data_norm) / (self.timebins * self.trials)

        return tot_covar

    def get_gauss_correlation(self, set_attr=True, **kwargs):
        """
        Computes the correlation matrix of the multivariate Gaussian that best fits the input binary spike trains.
        Inputs:
            :param set_attr: set to True to make computed correlation matrix an attribute of the class.
            :param kwargs: arguments for bisection algorithm method (see help(find_root_bisection)).

        Returns:
            :return: computed correlation matrix of multivariate Gaussian distribution.
        """
        data_mean = self.data.mean(1).mean(0)
        gauss_mean = self.gauss_mean
        if self.timebins > 1:
            data_covar = self.data_tvar_covariance
        else:
            data_covar = self.data_tfix_covariance

        gauss_corr = np.eye(self.num_neur)

        # Find pairwise correlation between each unique pair of neurons
        for i, j in zip(*self.tril_inds):
            # print("Neuron pair:", i, j)
            if np.abs(data_covar[i][j]) <= 1e-10:
                print('Data covariance is zero. Setting corresponding Gaussian dist. covariance to 0.')
                gauss_corr[i][j], gauss_corr[j][i] = 0., 0.

            else:
                x = find_root_bisection([data_mean[i], data_mean[j]],
                                        [gauss_mean[..., i], gauss_mean[..., j]],
                                        data_covar[i][j],
                                        **kwargs)
                gauss_corr[i][j], gauss_corr[j][i] = x, x

        if set_attr:
            setattr(self, 'gauss_corr', np.array(gauss_corr))
        return gauss_corr

    def _check_mean(self, mean):
        """Checks if input mean values lie between 0 and 1."""
        if np.any(mean < 0) or np.any(mean > 1):
            print('Mean should have value between 0 and 1.')
            raise NotImplementedError

if __name__ == "__main__":
    # test_dich_gauss()
    # plot_func_outputs()
    plot_dg_outputs()
