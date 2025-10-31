import pickle
import matplotlib.pyplot as plt
import numpy as np

temp_cipc = [[0.0 for j in range(10)] for i in range(10)]
temp_cipc[0][0] = 0.554
temp_cipc[0][1] = 0.566
temp_cipc[0][2] = 0.587
temp_cipc[0][3] = 0.560
temp_cipc[0][4] = 0.551
temp_cipc[0][5] = 0.599  # 0.013
temp_cipc[0][6] = 0.905
temp_cipc[0][7] = 0.928
temp_cipc[0][8] = 0.939
temp_cipc[0][9] = 0.975

temp_cipc[1][0] = 0.552
temp_cipc[1][1] = 0.576
temp_cipc[1][2] = 0.574
temp_cipc[1][3] = 0.588
temp_cipc[1][4] = 0.586
temp_cipc[1][5] = 0.572  # 0.013
temp_cipc[1][6] = 0.592
temp_cipc[1][7] = 0.592
temp_cipc[1][8] = 0.936
temp_cipc[1][9] = 0.969

temp_cipc[2][0] = 0.547
temp_cipc[2][1] = 0.561
temp_cipc[2][2] = 0.582
temp_cipc[2][3] = 0.570
temp_cipc[2][4] = 0.604
temp_cipc[2][5] = 0.593  # 0.013
temp_cipc[2][6] = 0.847
temp_cipc[2][7] = 0.928
temp_cipc[2][8] = 0.928
temp_cipc[2][9] = 0.989

temp_cipc[3][0] = 0.556
temp_cipc[3][1] = 0.570
temp_cipc[3][2] = 0.560
temp_cipc[3][3] = 0.600
temp_cipc[3][4] = 0.590
temp_cipc[3][5] = 0.569  # 0.013
temp_cipc[3][6] = 0.592
temp_cipc[3][7] = 0.597
temp_cipc[3][8] = 0.936
temp_cipc[3][9] = 0.980

temp_cipc[4][0] = 0.544
temp_cipc[4][1] = 0.520
temp_cipc[4][2] = 0.567
temp_cipc[4][3] = 0.584
temp_cipc[4][4] = 0.551
temp_cipc[4][5] = 0.594  # 0.013
temp_cipc[4][6] = 0.900
temp_cipc[4][7] = 0.921
temp_cipc[4][8] = 0.934
temp_cipc[4][9] = 0.975

temp_cipc[5][0] = 0.535
temp_cipc[5][1] = 0.544
temp_cipc[5][2] = 0.586
temp_cipc[5][3] = 0.587
temp_cipc[5][4] = 0.918
temp_cipc[5][5] = 0.910  # 0.013
temp_cipc[5][6] = 0.521
temp_cipc[5][7] = 0.901
temp_cipc[5][8] = 0.900
temp_cipc[5][9] = 0.921

temp_cipc[6][0] = 0.542
temp_cipc[6][1] = 0.573
temp_cipc[6][2] = 0.575
temp_cipc[6][3] = 0.591
temp_cipc[6][4] = 0.591
temp_cipc[6][5] = 0.601  # 0.013
temp_cipc[6][6] = 0.563
temp_cipc[6][7] = 0.927
temp_cipc[6][8] = 0.920
temp_cipc[6][9] = 0.939

temp_cipc[7][0] = 0.539
temp_cipc[7][1] = 0.554
temp_cipc[7][2] = 0.570
temp_cipc[7][3] = 0.588
temp_cipc[7][4] = 0.582
temp_cipc[7][5] = 0.602  # 0.013
temp_cipc[7][6] = 0.902
temp_cipc[7][7] = 0.611
temp_cipc[7][8] = 0.659
temp_cipc[7][9] = 0.937

temp_cipc[8][0] = 0.509
temp_cipc[8][1] = 0.548
temp_cipc[8][2] = 0.588
temp_cipc[8][3] = 0.594
temp_cipc[8][4] = 0.882
temp_cipc[8][5] = 0.569  # 0.013
temp_cipc[8][6] = 0.598
temp_cipc[8][7] = 0.597
temp_cipc[8][8] = 0.940
temp_cipc[8][9] = 0.933

temp_cipc[9][0] = 0.555
temp_cipc[9][1] = 0.568
temp_cipc[9][2] = 0.587
temp_cipc[9][3] = 0.582
temp_cipc[9][4] = 0.909
temp_cipc[9][5] = 0.549  # 0.013
temp_cipc[9][6] = 0.587
temp_cipc[9][7] = 0.933
temp_cipc[9][8] = 0.930
temp_cipc[9][9] = 0.937
R2_CPICs_old = np.array(temp_cipc)

num_init = 100

# collect the CPIC data.
R2_CPICs = list()
loss_CPIC = list()
saved_root = "res/lorenz_stochastic_infonce_exploration"
for i in range(num_init):
    with open(saved_root + "/latent_R2_seed{}.pkl".format(i), "rb") as f:
        res = pickle.load(f)
    R2_metrics = res["R2_metrics"]
    R2_CPICs.append(R2_metrics)
    losses = res["losses"]
    loss_CPIC.append(losses)
R2_CPICs = np.stack(R2_CPICs)[:,:,-1]
loss_CPIC = np.stack(loss_CPIC)

# modifies the certain R2 for some SNRs
# R2_CPICs[:, :4] = R2_CPICs_old[:, :4]
# R2_CPICs[:, -1] = R2_CPICs_old[:, -1]

R2_CPICs_mean = np.mean(R2_CPICs, axis=0)
R2_CPICs_std = np.std(R2_CPICs, axis=0)


# collect the CPIC data.
R2_CPICs_obs = list()
loss_CPICs_obs = list()
saved_root = "res/lorenz_stochastic_infonce_obs_exploration"
for i in range(num_init):
    with open(saved_root + "/latent_R2_seed{}.pkl".format(i), "rb") as f:
        res = pickle.load(f)
    R2_metrics = res["R2_metrics"]
    losses = res["losses"]
    R2_CPICs_obs.append(R2_metrics)
    loss_CPICs_obs.append(losses)
R2_CPICs_obs = np.stack(R2_CPICs_obs)[:,:,-1]
loss_CPICs_obs = np.stack(loss_CPICs_obs)

R2_CPICs_obs_mean = np.mean(R2_CPICs_obs, axis=0)
R2_CPICs_obs_std = np.std(R2_CPICs_obs, axis=0)

# collect the CPIC data.
R2_CPICs_det = list()
loss_CPICs_det = list()
saved_root = "res/lorenz_deterministic_infonce_exploration"
for i in range(num_init):
    with open(saved_root + "/latent_R2_seed{}.pkl".format(i), "rb") as f:
        res = pickle.load(f)
    R2_metrics = res["R2_metrics"]
    losses = res["losses"]
    R2_CPICs_det.append(R2_metrics)
    loss_CPICs_det.append(losses)
R2_CPICs_det = np.stack(R2_CPICs_det)[:,:,-1]
loss_CPICs_det = np.stack(loss_CPICs_det)

R2_CPICs_det_mean = np.mean(R2_CPICs_det, axis=0)
R2_CPICs_det_std = np.std(R2_CPICs_det, axis=0)

# collect the CPIC data.
R2_CPICs_det_obs = list()
loss_CPICs_det_obs = list()
saved_root = "res/lorenz_deterministic_infonce_obs_exploration"
for i in range(num_init):
    with open(saved_root + "/latent_R2_seed{}.pkl".format(i), "rb") as f:
        res = pickle.load(f)
    R2_metrics = res["R2_metrics"]
    losses = res["losses"]
    R2_CPICs_det_obs.append(R2_metrics)
    loss_CPICs_det_obs.append(losses)
R2_CPICs_det_obs = np.stack(R2_CPICs_det_obs)[:,:,-1]
loss_CPICs_det_obs = np.stack(loss_CPICs_det_obs)

R2_CPICs_det_obs_mean = np.mean(R2_CPICs_det_obs, axis=0)
R2_CPICs_det_obs_std = np.std(R2_CPICs_det_obs, axis=0)


if __name__ == "__main__":
    snr_vals = np.logspace(-3, -1, num=10)
    with open("res/lorenz_dca/latent_R2_{}.pkl".format(num_init), "rb") as f:
        res = pickle.load(f)

    R2_DCAs = res["R2_DCAs"]
    best_R2_DCAs = res["best_R2_DCAs"]
    fig = plt.figure()
    plt.plot(snr_vals, best_R2_DCAs)
    plt.scatter(snr_vals, best_R2_DCAs)
    plt.xscale('log')
    plt.show()

    R2_DCAs_mean = np.mean(R2_DCAs, axis=0)
    R2_DCAs_std = np.std(R2_DCAs, axis=0)

    fig = plt.figure(figsize=(5,5))
    plt.plot(snr_vals, R2_DCAs_mean, color="black", label="DCA")
    # plt.errorbar(snr_vals, R2_DCAs_mean, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_DCAs_std,
    #              c="black")
    plt.plot(snr_vals, R2_CPICs_mean, color="red", label="CPIC")
    # plt.errorbar(snr_vals, R2_CPICs_mean, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_CPICs_std,
    #              c="red")
    plt.plot(snr_vals, R2_CPICs_obs_mean, linestyle="dashed", color="red", label="CPIC_obs")
    # plt.errorbar(snr_vals, R2_CPICs_obs_mean, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_CPICs_obs_std,
    #              c="green")
    plt.plot(snr_vals, R2_CPICs_det_mean, color="blue", label="CPIC_det")
    # plt.errorbar(snr_vals, R2_CPICs_det_mean, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_CPICs_det_std,
    #              c="blue")
    plt.plot(snr_vals, R2_CPICs_det_obs_mean, linestyle="dashed", color="blue", label="CPIC_det_obs")
    # plt.errorbar(snr_vals, R2_CPICs_det_obs_mean, capsize=4, elinewidth=3, alpha=0.7, yerr=R2_CPICs_det_obs_std,
    #              c="yellow")
    plt.xscale('log')
    plt.xlabel('Signal-to-noise ratio (SNR)', fontsize=18)
    plt.ylabel('R\u00b2 regression score', fontsize=18)
    plt.yticks(np.array([0.4, 0.7, 1.0]), fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig/R2_lorenz.png")
    plt.show()
    