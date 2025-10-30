import h5py
from utils.data_util import compute_R2
from dca import DynamicalComponentsAnalysis as DCA
import numpy as np
import pickle
from tqdm import tqdm


if __name__ == "__main__":
    N = 100
    T = 4
    num_samples = 10000
    RESULTS_FILENAME = "data/lorenz/lorenz_exploration.hdf5"
    # load data
    with h5py.File(RESULTS_FILENAME, "r") as f:
        snr_vals = f.attrs["snr_vals"][:]
        X = f["X"][:]
        X_dynamics = f["X_dynamics"][:]
        X_noisy_dset = f["X_noisy"][:]
        X_pca_trans_dset = f["X_pca_trans"][:]
        X_dca_trans_dset = f["X_dca_trans"][:]

    X_dca_trans_dsets = np.zeros((N, len(snr_vals), num_samples, 3))
    R2_DCAs = np.zeros((N, len(snr_vals)))
    best_R2_DCAs = np.zeros(len(snr_vals))

    snr_idx = -1
    for snr_val, X_pca_trans, X_dca_trans, X_noisy in zip(snr_vals, X_pca_trans_dset, X_dca_trans_dset, X_noisy_dset):
        snr_idx += 1

        # reestimate DCA
        pis = list()
        coefs = list()
        for i in tqdm(range(N)):
            opt = DCA(T=T, d=3, n_init=1, rng_or_seed=i+1)
            opt.fit(X_noisy)
            pi = opt.score(X_noisy)
            coef = opt.coef_
            pis.append(pi)
            coefs.append(coef)
            #
            V_dca = coef
            # Project data onto DCA and PCA bases
            X_dca = np.dot(X_noisy, V_dca)
            beta_dca = np.linalg.lstsq(X_dca, X_dynamics, rcond=None)[0]
            X_dca_trans = np.dot(X_dca, beta_dca)
            # Save transformed projections
            X_dca_trans_dsets[i, snr_idx] = X_dca_trans
            R2_DCA = compute_R2(X_dca_trans, X_dynamics)
            R2_DCAs[i, snr_idx] = R2_DCA

        idx = np.argmax(pis)
        best_coefs = coefs[idx]
        best_R2_DCAs[snr_idx] = R2_DCAs[idx, snr_idx]

        print("SNR={}: R2DCA={}({})".format(snr_val, np.mean(R2_DCAs[:, snr_idx]), np.std(R2_DCAs[:, snr_idx])))
        print(R2_DCAs[:, snr_idx])
        # import pdb; pdb.set_trace()

    # save result
    with open("res/lorenz_dca/latent_R2_{}.pkl".format(N), "wb") as f:
        pickle.dump({"R2_DCAs": R2_DCAs, "best_R2_DCAs": best_R2_DCAs}, f)
    # import pdb; pdb.set_trace()
    