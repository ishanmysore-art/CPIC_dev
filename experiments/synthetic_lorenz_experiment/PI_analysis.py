import numpy as np
import pickle
from mi_estimator import GassianMIEstimator, MIEstimator, MIEstimator_Trainer, ddp_setup
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
from tqdm import tqdm

# Distributed Data Parallelization
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import destroy_process_group

critic_type = 'concat'  # or 'separable'
mi_params = dict(estimator='infonce', critic=critic_type, baseline='constant')
data_params = {
    'dim': 20,
    'batch_size': 1024,
    'rho': 0.7,
    'data_size': 10000
}
critic_params = {
    'n_layers': 2,
    'x_dim': 20,
    'embed_dim': 256,
    'y_dim': 20,
    'activation': 'relu',
}
opt_params = {
    'patience': 20,
    'min_delta': 0,
    'n_epochs': 1000,
    'learning_rate': 5e-4,
}


class JointData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])
        return sample


def create_pair_data(ts, T_past, T_future):
    nbatch, ndims, T = ts.shape
    T_joint = T_past + T_future
    ts_past = list()
    ts_future = list()
    for batch in range(nbatch):
        for i in range(T - T_joint + 1):
            ts_past.append(ts[batch, :, i:i+T_past].T.reshape(-1))
            ts_future.append(ts[batch, :, i + T_past: i + T_joint].T.reshape(-1))
    ts_past = np.stack(ts_past)
    ts_future = np.stack(ts_future)
    return ts_past, ts_future


def prepare_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle)

def prepare_paralleled_dataloader(dataset, batch_size):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      sampler=DistributedSampler(dataset))

def neural_PI_estimation(rank, world_size, ts, T_past, T_future, file_name):
    ddp_setup(rank, world_size)
    x_past, x_future = create_pair_data(ts, T_past, T_future)
    data_params['data_size'], data_params['dim'] = x_past.shape
    critic_params['x_dim'] = x_past.shape[1]
    critic_params['y_dim'] = x_future.shape[1]
    xy_dataset = JointData(x_past, x_future)
    # dataloader = prepare_dataloader(xy_dataset, batch_size=data_params["batch_size"], shuffle=True)
    dataloader = prepare_paralleled_dataloader(xy_dataset, batch_size=data_params["batch_size"])
    neural_mi = MIEstimator(critic_params, data_params, mi_params)
    # neural_mi_trainer = MIEstimator_Trainer(neural_mi, device='cuda', opt_params=opt_params, tqdm_disable=False)
    neural_mi_trainer = MIEstimator_Trainer(neural_mi, device=rank, opt_params=opt_params, tqdm_disable=False)
    pi_est = neural_mi_trainer.fit(dataloader, epochs=opt_params["n_epochs"], patience=opt_params["patience"],
                               min_delta=opt_params["min_delta"], show_history=True, save_file=file_name)
    destroy_process_group()

    return pi_est


def main(ts_list, T_past, T_future, do_gaussian_estimation=False, do_neural_estimation=False, experiment="sync1"):

    pi_est_gaussian_list = list()
    pi_est_neural_list = list()

    for i, ts in enumerate(ts_list):
        if do_gaussian_estimation:
            gaussian_estimator = GassianMIEstimator(T_past, T_future)
            pi_est_gaussian = gaussian_estimator.fit(ts)
            print("Predictive information (Gaussian) = {}".format(pi_est_gaussian))
            with open("{}_gaussian_pi_{}_{}_{}.pickle".format(experiment, T_past, T_future, i), "wb") as f:
                pickle.dump(pi_est_gaussian, f)
            pi_est_gaussian_list.append(pi_est_gaussian)

        if do_neural_estimation:
            world_size = torch.cuda.device_count()
            file_name = "{}_neural_pi_{}_{}_{}.pickle".format(experiment, T_past, T_future, i)
            mp.spawn(neural_PI_estimation, args=(world_size, ts, T_past, T_future, file_name), nprocs=world_size)
            with open("{}_neural_pi_{}_{}_{}.pickle".format(experiment, T_past, T_future, i), "rb") as file:
                pi_est_neural = pickle.load(file)

            pi_est_neural_list.append(pi_est_neural)
            print("Predictive information (Neural) = {}".format(pi_est_neural[1]))
    if do_gaussian_estimation:
        with open("{}_gaussian_pi_{}_{}.pickle".format(experiment, T_past, T_future), "wb") as f:
            pickle.dump(pi_est_gaussian_list, f)
    if do_neural_estimation:
        with open("{}_neural_pi_{}_{}.pickle".format(experiment, T_past, T_future), "wb") as f:
            pickle.dump(pi_est_neural_list, f)


if __name__ == "__main__":

    sync_version = 2

    if sync_version == 1:
        # PI estimation for synthetic signals (cosine signal + white noise)
        with open("syn_sin_ts.pickle", "rb") as file:
            ts_list = pickle.load(file)

        # T_list = [(2, 2), (5, 5), (10, 10), (20, 20), (50, 50), (100, 100)]
        T_list = [(2, 2), (5, 5), (10, 10), (20, 20), (50, 50), (100, 100), (200, 200), (500, 500)]
        for T_past, T_future in T_list:
            main(ts_list, T_past, T_future)
    elif sync_version == 2:
        # PI estimation for synthetic signals (GP signal)
        with open("syn_gp_ts.pickle", "rb") as file:
            X, ts_array, Sigmas = pickle.load(file) # X: (2000, 1); ts: (4, 2000, 100)

        T_list = [(10 * (i + 1), 10 * (i + 1)) for i in range(40)]

        # convert ts
        ts_list = [np.expand_dims(ts.transpose(1, 0), axis=1) for ts in ts_array]
        for T_past, T_future in tqdm(T_list):
            main(ts_list, T_past, T_future, experiment="sync2", do_neural_estimation=False, do_gaussian_estimation=True)

        import pdb; pdb.set_trace()
