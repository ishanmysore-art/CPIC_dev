import numpy as np
from torch.utils.data import Dataset


class PastFutureDataset(Dataset):
    def __init__(self, ts_list, window_size):
        """
        :param ts: a list of time series T_i x N
        :param window_size:
        """

        # if standardization:
        #     ts = (ts.T/ts.std(axis=1)).T
        past_ts = []
        future_ts = []
        for ts in ts_list:
            T, N = ts.shape
            for i in range(T-2*window_size):
                past_ts.append(ts[i:(i+window_size)])
                future_ts.append(ts[(i+window_size):(i+2*window_size)])
            self.past_ts = np.stack(past_ts)
            self.future_ts = np.stack(future_ts)

    def __len__(self):
        return len(self.past_ts)

    def __getitem__(self, idx):
        return self.past_ts[idx], self.future_ts[idx]
        