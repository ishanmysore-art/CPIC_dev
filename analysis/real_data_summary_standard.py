import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def summary_statistics(x, axis=0):
    return np.mean(x, axis=axis), np.std(x, axis=axis)


if __name__ == "__main__":

    m1_dims = np.array([5])
    hc_dims = np.array([5])
    temp_dims = np.array([5])
    ms_dims = np.array([5])

    for dim in m1_dims:
        with open("res/m1_stochastic_infonce_alt/result_dim{}_standard.pkl".format(dim), "rb") as f:
            m1_sto_infonce_R2s = pickle.load(f)
        m1_sto_infonce_R2s = np.squeeze(m1_sto_infonce_R2s)
        print("m1(stochastic), dim:{}, R2s:{}".format(dim, m1_sto_infonce_R2s))
        # with open("res/m1_deterministic_infonce_alt/result_dim{}_standard.pkl".format(dim), "rb") as f:
        #     m1_det_infonce_R2s = pickle.load(f)
        # m1_det_infonce_R2s = np.squeeze(m1_det_infonce_R2s)
        # m1_det_infonce_R2_mean, m1_det_infonce_R2_std = summary_statistics(m1_det_infonce_R2s, axis=0)
        # print("m1(deterministic), dim:{}, mean: {}, std: {}".format(dim, m1_det_infonce_R2_mean, m1_det_infonce_R2_std))
    for dim in hc_dims:
        with open("res/hc_stochastic_infonce_alt/result_dim{}_standard.pkl".format(dim), "rb") as f:
            hc_sto_infonce_R2s = pickle.load(f)
        hc_sto_infonce_R2s = np.squeeze(hc_sto_infonce_R2s)
        print("hc(stochastic), dim:{}, R2s:{}".format(dim, hc_sto_infonce_R2s))
        # with open("res/hc_deterministic_infonce_alt/result_dim{}_standard.pkl".format(dim), "rb") as f:
        #     hc_det_infonce_R2s = pickle.load(f)
        # hc_det_infonce_R2s = np.squeeze(hc_det_infonce_R2s)
        # hc_det_infonce_R2_mean, hc_det_infonce_R2_std = summary_statistics(hc_det_infonce_R2s, axis=0)
        # print("hc(deterministic), dim:{}, mean: {}, std: {}".format(dim, hc_det_infonce_R2_mean, hc_det_infonce_R2_std))
    # for dim in temp_dims:
    #     with open("res/temp_stochastic_infonce/result_dim{}_standard.pkl".format(dim), "rb") as f:
    #         temp_sto_infonce_R2s = pickle.load(f)
    #     temp_sto_infonce_R2s = np.squeeze(temp_sto_infonce_R2s)
    #     temp_sto_infonce_R2_mean, temp_sto_infonce_R2_std = summary_statistics(temp_sto_infonce_R2s, axis=0)
    #     print("temp(stochastic), dim:{}, mean: {}, std: {}".format(dim, temp_sto_infonce_R2_mean, temp_sto_infonce_R2_std))
    for dim in ms_dims:
        with open("res/ms_stochastic_infonce_alt/result_dim{}_standard.pkl".format(dim), "rb") as f:
            ms_sto_infonce_R2s = pickle.load(f)
        ms_sto_infonce_R2s = np.squeeze(ms_sto_infonce_R2s)
        print("ms(stochastic), dim:{}, R2s:{}".format(dim, ms_sto_infonce_R2s))


    # m1_dims = np.array([5, 10, 20, 30])
    # temp_dims = np.array([3, 4, 5, 6])
    # hc_dims = np.array([5, 10, 15, 25])
    # dims = 4


    import pdb; pdb.set_trace()

