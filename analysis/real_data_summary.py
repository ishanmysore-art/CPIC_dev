import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def summary_statistics(x, axis=0):
    return np.mean(x, axis=axis), np.std(x, axis=axis)


if __name__ == "__main__":
    do_suammrization_R2_CIPCs = False
    do_summarization_R2_CPIC_Baseline = False

    if do_suammrization_R2_CIPCs:
        # m1_dims = np.array([5, 10, 20, 30])
        m1_dims = np.array([5])
        # hc_dims = np.array([5, 10, 15, 20])
        hc_dims = np.array([5])
        # temp_dims = np.array([3, 4, 5, 6])
        # temp_dims = np.array([5])

        for dim in m1_dims:
            with open("res/m1_stochastic_infonce_alt/result_dim{}.pkl".format(dim), "rb") as f:
                m1_sto_infonce_R2s = pickle.load(f)
            m1_sto_infonce_R2s = np.squeeze(m1_sto_infonce_R2s)
            m1_sto_infonce_R2_mean, m1_sto_infonce_R2_std = summary_statistics(m1_sto_infonce_R2s, axis=0)
            print("m1(stochastic), dim:{}, mean: {}, std: {}".format(dim, m1_sto_infonce_R2_mean, m1_sto_infonce_R2_std))
            with open("res/m1_deterministic_infonce_alt/result_dim{}.pkl".format(dim), "rb") as f:
                m1_det_infonce_R2s = pickle.load(f)
            m1_det_infonce_R2s = np.squeeze(m1_det_infonce_R2s)
            m1_det_infonce_R2_mean, m1_det_infonce_R2_std = summary_statistics(m1_det_infonce_R2s, axis=0)
            print("m1(deterministic), dim:{}, mean: {}, std: {}".format(dim, m1_det_infonce_R2_mean, m1_det_infonce_R2_std))
        for dim in hc_dims:
            with open("res/hc_stochastic_infonce_alt/result_dim{}.pkl".format(dim), "rb") as f:
                hc_sto_infonce_R2s = pickle.load(f)
            hc_sto_infonce_R2s = np.squeeze(hc_sto_infonce_R2s)
            hc_sto_infonce_R2_mean, hc_sto_infonce_R2_std = summary_statistics(hc_sto_infonce_R2s, axis=0)
            print("hc(stochastic), dim:{}, mean: {}, std: {}".format(dim, hc_sto_infonce_R2_mean, hc_sto_infonce_R2_std))
            with open("res/hc_deterministic_infonce_alt/result_dim{}.pkl".format(dim), "rb") as f:
                hc_det_infonce_R2s = pickle.load(f)
            hc_det_infonce_R2s = np.squeeze(hc_det_infonce_R2s)
            hc_det_infonce_R2_mean, hc_det_infonce_R2_std = summary_statistics(hc_det_infonce_R2s, axis=0)
            print("hc(deterministic), dim:{}, mean: {}, std: {}".format(dim, hc_det_infonce_R2_mean, hc_det_infonce_R2_std))
        # for dim in temp_dims:
        #     with open("res/temp_stochastic_infonce/result_dim{}.pkl".format(dim), "rb") as f:
        #         temp_sto_infonce_R2s = pickle.load(f)
        #     temp_sto_infonce_R2s = np.squeeze(temp_sto_infonce_R2s)
        #     temp_sto_infonce_R2_mean, temp_sto_infonce_R2_std = summary_statistics(temp_sto_infonce_R2s, axis=0)
        #     print("temp(stochastic), dim:{}, mean: {}, std: {}".format(dim, temp_sto_infonce_R2_mean, temp_sto_infonce_R2_std))

        # m1_dims = np.array([5, 10, 20, 30])
        # temp_dims = np.array([3, 4, 5, 6])
        # hc_dims = np.array([5, 10, 15, 25])
        # dims = 4

    if do_summarization_R2_CPIC_Baseline:
        # Visualize the R2 improvements
        m1_dims = np.array([5, 10, 20, 30])
        hc_dims = np.array([5, 10, 15, 25])
        dims = 1
        Ts = [3,4,5,6]

        with open("res/m1_stochastic_infonce/result_cpt.pkl", "rb") as f:
            m1_results = pickle.load(f)
        m1_results = m1_results.transpose(0,1,3,2)
        m1_resutls_mean = np.mean(m1_results, axis=0)
        m1_resutls_std = np.std(m1_results, axis=0)
        for i in range(dims):
            print("m1, dim:{}, mean: {}, std: {}".format(m1_dims[i], m1_resutls_mean[i], m1_resutls_std[i]))

        with open("res/hc_stochastic_infonce/result_cpt.pkl", "rb") as f:
            hc_results = pickle.load(f)
        hc_results = hc_results.transpose(0,1,3,2)
        hc_resutls_mean = np.mean(hc_results, axis=0)
        hc_resutls_std = np.std(hc_results, axis=0)
        for i in range(dims):
            print("hc, dim:{}, mean: {}, std: {}".format(hc_dims[i], hc_resutls_mean[i], hc_resutls_std[i]))

        # with open("res/temp_stochastic_infonce/result_cpt.pkl", "rb") as f:
        #     temp_results = pickle.load(f)
        # temp_results = temp_results.transpose(0,1,3,2)
        # temp_resutls_mean = np.mean(temp_results, axis=0)
        # temp_resutls_std = np.std(temp_results, axis=0)
        # for i in range(dims):
        #     print("temp, dim:{}, mean: {}, std: {}".format(temp_dims[i], temp_resutls_mean[i], temp_resutls_std[i]))

        # plot R2 improve over PCA fro three forecasting task, each with three different lag values. Left: dorsal hipcampus.
        # Right: motor cortex
        colors = [matplotlib.cm.rainbow(x) for x in np.linspace(0, 1, 4)]
        fig = plt.figure(figsize=(6,4))
        handles = []
        lags = np.array([5, 10, 15])
        pca_plot, = plt.plot(lags, hc_resutls_mean[0][0] - hc_resutls_mean[0][0], color='b')
        # plt.plot(lags, hc_resutls_mean[0][1] - hc_resutls_mean[0][0], label="SFA")
        dca3_plot, = plt.plot(lags, hc_resutls_mean[0][2] - hc_resutls_mean[0][0], color=colors[0])
        dca4_plot, = plt.plot(lags, hc_resutls_mean[0][3] - hc_resutls_mean[0][0], color=colors[1])
        dca5_plot, = plt.plot(lags, hc_resutls_mean[0][4] - hc_resutls_mean[0][0], color=colors[2])
        dca6_plot, = plt.plot(lags, hc_resutls_mean[0][5] - hc_resutls_mean[0][0], color=colors[3])
        handles = [pca_plot, dca3_plot, dca4_plot, dca5_plot, dca6_plot]
        with open("res/hc_stochastic_infonce_alt/result_dim{}.pkl".format(5), "rb") as f:
            hc_sto_infonce_R2s = pickle.load(f)
        hc_sto_infonce_R2_mean, hc_sto_infonce_R2_std = summary_statistics(hc_sto_infonce_R2s, axis=0)
        for i in np.arange(4):
            temp = plt.plot(lags, hc_sto_infonce_R2_mean[0, :, i] - hc_resutls_mean[0][0], linestyle="dashed", color=colors[i])
            handles.append(temp[0])
        plt.xlabel("Lag", fontsize=15)
        plt.ylabel(r'$\Delta R^2$', fontsize=15)
        plt.xticks(ticks=lags, fontsize=15)
        plt.yticks(ticks=np.array([0.00, 0.05, 0.10]), fontsize=15)
        plt.tight_layout()
        plt.savefig("fig/hc_r2.png")
        plt.show()

        # Left: dorsal hippocampus
        fig = plt.figure(figsize=(6,4))
        lags = np.array([5, 10, 15])
        plt.plot(lags, m1_resutls_mean[0][0] - m1_resutls_mean[0][0], color='b')
        # plt.plot(lags, m1_resutls_mean[0][1] - m1_resutls_mean[0][0], label="SFA")
        plt.plot(lags, m1_resutls_mean[0][2] - m1_resutls_mean[0][0], color=colors[0])
        plt.plot(lags, m1_resutls_mean[0][3] - m1_resutls_mean[0][0], color=colors[1])
        plt.plot(lags, m1_resutls_mean[0][4] - m1_resutls_mean[0][0], color=colors[2])
        plt.plot(lags, m1_resutls_mean[0][5] - m1_resutls_mean[0][0], color=colors[3])
        with open("res/m1_stochastic_infonce_alt/result_dim{}.pkl".format(5), "rb") as f:
            m1_sto_infonce_R2s = pickle.load(f)

        m1_sto_infonce_R2_mean,  m1_sto_infonce_R2_std = summary_statistics(m1_sto_infonce_R2s, axis=0)
        for i in np.arange(4):
            temp = plt.plot(lags, m1_sto_infonce_R2_mean[0, :, i] - m1_resutls_mean[0][0], linestyle="dashed", color=colors[i])
        plt.xlabel("Lag", fontsize=15)
        plt.ylabel(r'$\Delta R^2$', fontsize=15)
        plt.xticks(ticks=lags, fontsize=15)
        plt.yticks(ticks=np.array([0.00, 0.05, 0.10]), fontsize=15)
        plt.tight_layout()
        plt.savefig("fig/m1_r2.png")
        fig.show()

        figlegend = plt.figure(figsize=(2,4))
        figlegend.legend(handles, [r"$PCA$", r"$DCA_3$", r"$DCA_4$", r"$DCA_5$", r"$DCA_6$", r"$CPIC_3$", r"$CPIC_4$", r"$CPIC_5$", r"$CPIC_6$"], loc='center')
        plt.savefig("fig/m1_hc_legend.png")
        figlegend.show()

        # fig = plt.figure()
        # lags = np.array([5, 10, 15])
        # plt.plot(lags, m1_resutls_mean[0][0] - m1_resutls_mean[0][0], label="PCA")
        # plt.plot(lags, m1_resutls_mean[0][1] - m1_resutls_mean[0][0], label="SFA")
        # plt.plot(lags, m1_resutls_mean[0][2] - m1_resutls_mean[0][0], label="DCA")
        # with open("res/m1_stochastic_infonce_alt/result_dim{}.pkl".format(5), "rb") as f:
        #     m1_sto_infonce_R2s = pickle.load(f)
        # m1_sto_infonce_R2s = np.squeeze(m1_sto_infonce_R2s)
        # m1_sto_infonce_R2_mean, m1_sto_infonce_R2_std = summary_statistics(m1_sto_infonce_R2s, axis=0)
        # plt.plot(lags, m1_sto_infonce_R2_mean - m1_resutls_mean[0][0], label="CPIC")
        # # plt.legend(fontsize=15)
        # plt.xlabel("Lag", fontsize=15)
        # plt.ylabel(r'$\Delta R^2$', fontsize=15)
        # plt.xticks(ticks=lags, fontsize=15)
        # plt.yticks(ticks=np.array([0.00, 0.05, 0.10]), fontsize=15)
        # plt.savefig("fig/m1_r2_alt.png")
        # plt.show()
    import pdb; pdb.set_trace()

