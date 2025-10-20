import numpy as np
import pickle
import matplotlib.pyplot as plt
import sdepy
from sdepy import *  # safe and handy for interactive sessions
import os

# Gaussian Process package
import sklearn
from sklearn.gaussian_process.kernels import RBF
np.random.seed(42)


def ou_process_generator(theta=1., k=1., sigma=1., T=10000, xlim=(0, 1), do_visual=True):
    @sdepy.integrate
    def my_process(t, x, theta=theta, k=k, sigma=sigma):
        return {'dt': k * (theta - x), 'dw': sigma}

    timeline = np.linspace(*xlim, T)
    x = my_process(x0=0, paths=1)(timeline)
    if do_visual:
        fig = plt.figure()
        plt.plot(timeline, x)
        plt.show()
        plt.close(fig)
    return x


def plot_kernel(X, y, xlim, scatter=False):
    if scatter:
        for i in range(y.shape[1]):
             plt.scatter(X, y[:,i], alpha=0.8, s=3)
    else:
        for i in range(y.shape[1]):
             plt.plot(X, y[:,i], alpha=0.8)
    plt.ylabel('$y$', fontsize=13, labelpad=0)
    plt.xlabel('$x$', fontsize=13, labelpad=0)
    plt.xlim(xlim)


def gp_process_generator(T=100, n=100, amplitude=1., length_scale=1., xlim=(0, 1), do_plot=True):
    """
    :param T: number of time stamps
    :param n: number of realizations
    :return:
    """
    X = np.expand_dims(np.linspace(*xlim, T), 1)

    if do_plot:
        fig = plt.figure()

    kernel = RBF(length_scale=length_scale)
    Sigma = amplitude * kernel(X, X)
    Sigma = Sigma + np.eye(T) * 0.00001
    y = np.random.multivariate_normal(
        mean=np.zeros(T), cov=Sigma,
        size=n).T

    if do_plot:
        plot_kernel(X, y, xlim)
        fig.tight_layout()
        plt.show()
    return X, y, Sigma


def Compute_true_PI(Sigmas, T_list):
    PIs_list = list()
    for Sigma in Sigmas:
        T_full, T_full = Sigma.shape
        T = int(T_full/2)
        PIs = list()
        for t in T_list:
            Sigma_t = Sigma[:t, :t]
            Sigma_2t = Sigma[:2*t, :2*t]
            logdet_Sigma_t = np.linalg.slogdet(Sigma_t)[0] * np.linalg.slogdet(Sigma_t)[1]
            logdet_Sigma_2t = np.linalg.slogdet(Sigma_2t)[0] * np.linalg.slogdet(Sigma_2t)[1]
            print(logdet_Sigma_t, logdet_Sigma_2t)
            PI = 0.5*(2*logdet_Sigma_t - logdet_Sigma_2t)
            PIs.append(PI)
        PIs = np.array(PIs)
        PIs_list.append(PIs)
    return np.stack(PIs_list)


def data_generator(para_list, T, num_sample=1, random_seed=22, type="sin", xlim=(0, 1)):
    """
    :param para_list: a list of parameter pairs (amplitude, frequency)
    :param T: number of time steps
    :param num_sample: number of samples
    :return: time series samples
    """
    T_full = 2 * T
    np.random.seed(random_seed)
    time_stamps = np.arange(T_full)
    if type == "sin":
        data_list = list()
        for A, f in para_list:
            signal = A * np.sin(2 * np.pi * f * time_stamps)
            noise = np.random.randn(num_sample, T_full)
            data = signal + noise
            data_list.append(data)
        return np.stack(data_list)
    elif type == "ou":
        data_list = list()
        for theta, k, sigma in para_list:
            X, data, Sigma = ou_process_generator(theta, k, sigma, T_full)
            data_list.append(data)
        return np.stack(data_list)
    elif type == "gp":
        data_list = list()
        Sigma_list = list()
        for amplitude, length_scale in para_list:
            X, data, Sigma = gp_process_generator(T_full, num_sample, amplitude, length_scale, xlim=xlim)
            data_list.append(data)
            Sigma_list.append(Sigma)
        return X, np.stack(data_list), np.stack(Sigma_list)


def visualization(para_list, ts):
    for para, data in zip(para_list, ts):
        plt.plot(data.T)
        plt.title("A={}, f={}".format(*para))
        plt.show()


def plot_PI(T_list, PIs, lengthscales, save_file="figs/PI_vs_T.png"):
    Ts = np.array(T_list)
    fig = plt.figure()
    for i, lengthscale in enumerate(lengthscales):
        plt.plot(Ts, PIs[i], label="lengthscale = {}".format(lengthscale))
    plt.legend(fontsize=18)
    plt.xlabel("T", fontsize=20)
    plt.xticks(fontsize=16)
    plt.ylabel("PI", fontsize=20)
    plt.yticks(fontsize=16)
    plt.savefig(save_file)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # para_list = [(1, 1), (1, 5), (1, 20), (2, 1), (2, 5), (2, 20), (5, 1), (5, 5), (5, 20)]
    # ts = data_generator(para_list, T=10000, type="sin")
    # with open("syn_sin_ts.pickle", "wb") as file:
    #     pickle.dump(ts, file)
    # visualization(para_list, ts)

    # para_list=[(0, 0.01, 1), (0, 0.1, 1), (0, 1, 1)]
    # ts = data_generator(para_list, T=1000, type="ou")
    # with open("syn_ou_ts.pickle", "wb") as file:
    #     pickle.dump(ts, file)

    para_list = [(1, 0.1), (1, 0.2), (1, 0.3), (1, 0.4)]
    lengthscales = np.array([para[1] for para in para_list])

    if not os.path.exists("syn_gp_ts.pickle"):
        X, ts, Sigmas = data_generator(para_list, T=1000, num_sample=100, type="gp", xlim=(0, 10))
        with open("syn_gp_ts.pickle", "wb") as file:
            pickle.dump((X, ts, Sigmas), file)
    else:
        with open("syn_gp_ts.pickle", "rb") as file:
            X, ts, Sigmas = pickle.load(file)

    T_list = [10 * (i + 1) for i in range(40)]

    PIs = Compute_true_PI(Sigmas, T_list)
    print(PIs)
    plot_PI(T_list, PIs, lengthscales)


    import pdb; pdb.set_trace()


