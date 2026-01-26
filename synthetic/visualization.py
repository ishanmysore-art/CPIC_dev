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

def collect_data2(saved_root, num_init):
    R2_CPICs = list()
    loss_CPICs = list()
    for i in range(num_init):
        with open(saved_root + "/latent_R2.pkl", "rb") as f:
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
    num_init = 100
    snr_vals = np.logspace(-3, -1, num=10)

    with open("res/lorenz_dca/latent_R2_{}.pkl".format(num_init), "rb") as f:
        res = pickle.load(f)
    R2_DCAs = res["R2_DCAs"]
    best_R2_DCAs = res["best_R2_DCAs"]
    R2_DCAs_mean = np.mean(R2_DCAs, axis=0)
    R2_DCAs_std = np.std(R2_DCAs, axis=0)


    # collect the CPIC data.
    R2_CPICs_mean, R2_CPICs_std, R2_CPICs_opt = collect_data("res/lorenz_stochastic_infonce_exploration", 52)

    R2_CPICs_obs_mean, R2_CPICs_obs_std, R2_CPICs_obs_opt = collect_data2("res/lorenz_stochastic_infonce_obs_exploration", 1)

    R2_CPICs_det_mean, R2_CPICs_det_std, R2_CPICs_det_opt = collect_data("res/lorenz_deterministic_infonce_exploration", 86)
    R2_CPICs_det_obs_mean, R2_CPICs_det_obs_std, R2_CPICs_det_obs_opt = collect_data("res/lorenz_deterministic_infonce_obs_exploration", 57)


    fig = plt.figure(figsize=(5,5))
    plt.plot(snr_vals, best_R2_DCAs, color="black", label="DCA")
    plt.plot(snr_vals, R2_CPICs_obs_opt, color="red", linestyle="dashed", label="Stochastic CPIC(O))")
    plt.plot(snr_vals, R2_CPICs_opt, color="red", label="Stochastic CPIC(L)")
    plt.plot(snr_vals, R2_CPICs_det_obs_opt, color="blue", linestyle="dashed", label="Deterministic CPIC(O)")
    plt.plot(snr_vals, R2_CPICs_det_opt, color="blue", label="Deterministic CPIC(L)")

    plt.xscale('log')
    plt.xlabel('Signal-to-noise ratio (SNR)', fontsize=18)
    plt.ylabel('R\u00b2 regression score', fontsize=18)
    plt.yticks(np.array([0.4, 0.7, 1.0]), fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig/R2_lorenz_best_{}.png".format(num_init))
    plt.show()


    fig = plt.figure(figsize=(5,5))
    plt.plot(snr_vals, R2_DCAs_mean, color="black", label="DCA")
    plt.errorbar(snr_vals, R2_DCAs_mean, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_DCAs_std, c="black")
    
    plt.plot(snr_vals, R2_CPICs_obs_mean, linestyle="dashed", color="red", label="Stochastic CPIC(O)")
    plt.errorbar(snr_vals, R2_CPICs_obs_mean, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_CPICs_obs_std, c="red")
    
    plt.plot(snr_vals, R2_CPICs_mean, color="red", label="Stochastic CPIC(L)")
    plt.errorbar(snr_vals, R2_CPICs_mean, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_CPICs_std, c="red")
    
    plt.plot(snr_vals, R2_CPICs_det_obs_mean, linestyle="dashed", color="blue", label="Deterministic CPIC(O)")
    plt.errorbar(snr_vals, R2_CPICs_det_obs_mean, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_CPICs_det_obs_std, c="blue")
    
    plt.plot(snr_vals, R2_CPICs_det_mean, color="blue", label="Deterministic CPIC(L)")
    plt.errorbar(snr_vals, R2_CPICs_det_mean, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_CPICs_det_std, c="blue")
    
    plt.xscale('log')
    plt.xlabel('Signal-to-noise ratio (SNR)', fontsize=18)
    plt.ylabel('R\u00b2 regression score', fontsize=18)
    plt.yticks(np.array([0.4, 0.7, 1.0]), fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig/R2_lorenz_mean_{}.png".format(num_init))
    plt.show()
