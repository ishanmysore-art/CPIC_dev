import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np


def collect_data(saved_root, num_init):
    R2_CPICs = list()
    loss_CPICs = list()
    for i in range(num_init):
        with open(saved_root + "/latent_R2_seed{}.pkl".format(i), "rb") as f:
            res = pickle.load(f)
        R2_metrics = res["R2_metrics"]
        R2_CPICs.append(R2_metrics)
        losses = res["losses"]
        loss_CPICs.append(losses)
    R2_CPICs = np.stack(R2_CPICs)[:,:,-1]
    loss_CPICs = np.stack(loss_CPICs)

    R2_CPICs_mean = np.mean(R2_CPICs, axis=0)
    R2_CPICs_std = np.std(R2_CPICs, axis=0)
    R2_CPICs_opt = list()
    for idx, idx_min in enumerate(np.argmin(loss_CPICs, axis=0)):
        R2_CPICs_opt.append(R2_CPICs[idx_min, idx])

    return R2_CPICs_mean, R2_CPICs_std, R2_CPICs_opt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize synthetic CPIC results.")
    parser.add_argument(
        "--num_init",
        type=int,
        default=100,
        help="Number of initialization seeds to aggregate (default: 100)",
    )
    args = parser.parse_args()
    num_init = args.num_init
    snr_vals = np.logspace(-3, -1, num=10)

    # with open("res/lorenz_dca/latent_R2_{}.pkl".format(num_init), "rb") as f:
    #     res = pickle.load(f)
    # R2_DCAs = res["R2_DCAs"]
    # best_R2_DCAs = res["best_R2_DCAs"]
    # R2_DCAs_mean = np.mean(R2_DCAs, axis=0)
    # R2_DCAs_std = np.std(R2_DCAs, axis=0)


    # collect the CPIC results.
    R2_CPICs_mean, R2_CPICs_std, R2_CPICs_opt = collect_data("res/lorenz_stochastic_infonce_exploration", num_init)
    R2_CPICs_obs_mean, R2_CPICs_obs_std, R2_CPICs_obs_opt = collect_data("res/lorenz_stochastic_infonce_obs_exploration", num_init)
    R2_CPICs_det_mean, R2_CPICs_det_std, R2_CPICs_det_opt = collect_data("res/lorenz_deterministic_infonce_exploration", num_init)
    R2_CPICs_det_obs_mean, R2_CPICs_det_obs_std, R2_CPICs_det_obs_opt = collect_data("res/lorenz_deterministic_infonce_obs_exploration", num_init)
    
    # collect the CPIC conv results
    R2_CPICs_obs_mean_conv_s, R2_CPICs_obs_std_conv_s, R2_CPICs_obs_opt_conv_s = collect_data("res/lorenz_stochastic_infonce_obs_exploration_conv_s", num_init)
    R2_CPICs_obs_mean_conv_st, R2_CPICs_obs_std_conv_st, R2_CPICs_obs_opt_conv_st = collect_data("res/lorenz_stochastic_infonce_obs_exploration_conv_st", num_init)
    R2_CPICs_obs_mean_conv_t, R2_CPICs_obs_std_conv_t, R2_CPICs_obs_opt_conv_t = collect_data("res/lorenz_stochastic_infonce_obs_exploration_conv_t", num_init)


    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    # ax.plot(snr_vals, best_R2_DCAs, color="black", label="DCA")
    ax.plot(snr_vals, R2_CPICs_obs_opt, color="red", linestyle="dashed", label="Stochastic CPIC(O)")
    ax.plot(snr_vals, R2_CPICs_opt, color="red", label="Stochastic CPIC(L)")
    ax.plot(snr_vals, R2_CPICs_det_obs_opt, color="blue", linestyle="dashed", label="Deterministic CPIC(O)")
    ax.plot(snr_vals, R2_CPICs_det_opt, color="blue", label="Deterministic CPIC(L)")
    ax.plot(snr_vals, R2_CPICs_obs_opt_conv_s, color="green", linestyle="dashed", label="Stochastic CPIC(O, ConvSpatial)")
    ax.plot(snr_vals, R2_CPICs_obs_opt_conv_st, color="orange", linestyle="dashed", label="Stochastic CPIC(O, ConvSpatiotemporal)")
    ax.plot(snr_vals, R2_CPICs_obs_opt_conv_t, color="purple", linestyle="dashed", label="Stochastic CPIC(O, ConvTemporal)")

    ax.set_xscale('log')
    ax.set_xlabel('Signal-to-noise ratio (SNR)', fontsize=18)
    ax.set_ylabel('R\u00b2 regression score', fontsize=18)
    ax.set_yticks(np.array([0.4, 0.7, 1.0]))
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.subplots_adjust(right=0.58)
    plt.savefig("fig/R2_lorenz_best_{}.png".format(num_init), bbox_inches="tight")
    plt.show()


    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    # ax.plot(snr_vals, R2_DCAs_mean, color="black", label="DCA")
    # ax.errorbar(snr_vals, R2_DCAs_mean, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_DCAs_std, c="black")
    ax.plot(snr_vals, R2_CPICs_obs_mean, linestyle="dashed", color="red", label="Stochastic CPIC(O)")
    # ax.errorbar(snr_vals, R2_CPICs_obs_mean, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_CPICs_obs_std, c="red")
    ax.plot(snr_vals, R2_CPICs_mean, color="red", label="Stochastic CPIC(L)")
    # ax.errorbar(snr_vals, R2_CPICs_mean, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_CPICs_std, c="red")
    ax.plot(snr_vals, R2_CPICs_det_obs_mean, linestyle="dashed", color="blue", label="Deterministic CPIC(O)")
    # ax.errorbar(snr_vals, R2_CPICs_det_obs_mean, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_CPICs_det_obs_std, c="blue")
    ax.plot(snr_vals, R2_CPICs_det_mean, color="blue", label="Deterministic CPIC(L)")
    # ax.errorbar(snr_vals, R2_CPICs_det_mean, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_CPICs_det_std, c="blue")
    ax.plot(snr_vals, R2_CPICs_obs_mean_conv_s, linestyle="dashed", color="green", label="Stochastic CPIC(O, ConvSpatial)")
    # ax.errorbar(snr_vals, R2_CPICs_obs_mean_conv_s, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_CPICs_obs_std_conv_s, c="green")
    ax.plot(snr_vals, R2_CPICs_obs_mean_conv_st, linestyle="dashed", color="orange", label="Stochastic CPIC(O, ConvSpatiotemporal)")
    # ax.errorbar(snr_vals, R2_CPICs_obs_mean_conv_st, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_CPICs_obs_std_conv_st, c="orange")
    ax.plot(snr_vals, R2_CPICs_obs_mean_conv_t, linestyle="dashed", color="purple", label="Stochastic CPIC(O, ConvTemporal)")
    # ax.errorbar(snr_vals, R2_CPICs_obs_mean_conv_t, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_CPICs_obs_std_conv_t, c="purple")

    ax.set_xscale('log')
    ax.set_xlabel('Signal-to-noise ratio (SNR)', fontsize=18)
    ax.set_ylabel('R\u00b2 regression score', fontsize=18)
    ax.set_yticks(np.array([0.4, 0.7, 1.0]))
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.subplots_adjust(right=0.58)
    plt.savefig("fig/R2_lorenz_mean_{}.png".format(num_init), bbox_inches="tight")
    plt.show()
