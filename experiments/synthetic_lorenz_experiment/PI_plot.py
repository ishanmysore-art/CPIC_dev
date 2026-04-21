import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_pi_sync1(pis_mat, save_file="sync1_PI_vs_T_gaussian.png", title=None):
    plt.figure()
    for y, para, col, ls in zip(pis_mat, para_list, color_list, linestyle_list):
        plt.plot(Ts, y, label="A={}, f={}".format(*para), color=col, linestyle=ls)
    plt.xticks(Ts, Ts)
    plt.legend()
    plt.title(title)
    plt.savefig(save_file)
    plt.show()

def plot_pi_sync2(pis_mat, save_file="sync2_PI_vs_T_gaussian.png", title=None):
    plt.figure()
    for y, para, col, ls in zip(pis_mat, para_list, color_list, linestyle_list):
        plt.plot(Ts, y, label="length-scale={}".format(para), color=col, linestyle=ls)
    plt.xticks(Ts, Ts)
    plt.legend()
    plt.title(title)
    plt.savefig(save_file)
    plt.show()


if __name__ == "__main__":

    sync_version = 2

    # methods = ["gaussian", "neural"]
    methods = ["gaussian"]

    if sync_version == 1:
        Ts = [2, 5, 10, 20, 50, 100, 200, 500]
        para_list = [(1, 1), (1, 5), (1, 20), (2, 1), (2, 5), (2, 20), (5, 1), (5, 5), (5, 20)]
        color_list = ['r', 'r', 'r', 'b', 'b', 'b', 'g', 'g', 'g']
        linestyle_list = ['solid', 'dotted', 'dashed', 'solid', 'dotted', 'dashed', 'solid', 'dotted', 'dashed']
        methods = ["gaussian", "neural"]
        for method in methods:
            pis_list = list()
            for data_index in range(9):
                pi_list = list()
                for T in Ts:
                    with open("{}_pi_{}_{}_{}.pickle".format(method, T, T, data_index), "rb") as f:
                        res = pickle.load(f)
                    if method == "gaussian":
                        pi_list.append(res)
                    elif method == "neural":
                        pi_list.append(res[0][-1])
                    else:
                        raise NotImplementedError("{} is not implemented.".format(method))
                pis = np.array(pi_list)
                pis_list.append(pis)
            pis_mat = np.stack(pis_list)
            plot_pi_sync1(pis_mat, save_file="sync1_PI_vs_T_{}.png".format(method), title="PI vs T ({})".format(method))
        import pdb; pdb.set_trace()
    elif sync_version == 2:
        experiment = "sync2"
        T_list = [(10 * (i + 1), 10 * (i + 1)) for i in range(40)]
        Ts = [10 * (i+1) for i in range(40)]
        para_list = [(0.1), (0.2), (0.3), (0.4)]
        color_list = ['r', 'b', 'g', 'k']
        linestyle_list = ['solid',  'solid', 'solid', 'solid']
        for method in methods:
            pis_list = list()
            for data_index in range(4):
                pi_list = list()
                for T_past, T_future in T_list:
                    with open("{}_{}_pi_{}_{}_{}.pickle".format(experiment, method, T_past, T_future, data_index), "rb") as f:
                        res = pickle.load(f)
                    if method == "gaussian":
                        pi_list.append(res)
                    elif method == "neural":
                        pi_list.append(res[0][-1])
                    else:
                        raise NotImplementedError("{} is not implemented.".format(method))
                pis = np.array(pi_list)
                pis_list.append(pis)
            pis_mat = np.stack(pis_list)
            # plot_pi_sync2(pis_mat, save_file="sync2_PI_vs_T_{}.png".format(method), title="PI vs T ({})".format(method))
            plot_pi_sync2(pis_mat, save_file="sync2_PI_vs_T_{}.png".format(method), title="")
