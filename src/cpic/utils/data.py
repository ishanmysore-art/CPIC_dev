import numpy as np
from torch.utils.data import Dataset


class PastFutureDataset(Dataset):
    """
    Dataset for past and future time series data.

    Parameters
    ----------
    ts_list : list of numpy.ndarray
        List of time series arrays, each of shape (T_i, N),
        where T_i is the length of the series and N is the number of variables.
    window_size : int
        Length of the past and future windows (in time steps).

    Attributes
    ----------
    past_ts : numpy.ndarray of shape (num_windows, window_size, N)
        Past time series data.
    future_ts : numpy.ndarray of shape (num_windows, window_size, N)
        Future time series data.
    """

    def __init__(self, ts_list, window_size):
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
        