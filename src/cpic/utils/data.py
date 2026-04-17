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

    Notes
    -----
    Windows are built lazily in ``__getitem__`` so memory stays O(sum_i T_i * N)
    instead of materializing all (past, future) pairs (which was O(num_windows * W * N)).
    """

    def __init__(self, ts_list, window_size):
        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")

        self.ts_list = [np.asarray(ts) for ts in ts_list]
        self._segments = []
        cum = 0
        for ts in self.ts_list:
            T, N = ts.shape
            nw = T - 2 * self.window_size
            if nw <= 0:
                raise ValueError(
                    f"Each series needs length > 2*window_size; got T={T}, window_size={self.window_size}"
                )
            self._segments.append({"ts": ts, "base": cum, "nw": nw})
            cum += nw
        self._len = cum

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if idx < 0 or idx >= self._len:
            raise IndexError(idx)
        W = self.window_size
        for seg in self._segments:
            if idx < seg["base"] + seg["nw"]:
                i = idx - seg["base"]
                ts = seg["ts"]
                return ts[i : i + W], ts[i + W : i + 2 * W]
        raise IndexError(idx)
