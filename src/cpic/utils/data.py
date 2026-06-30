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


class InputOutputPastFutureDataset(Dataset):
    """
    Dataset for multi-input / single-output (MISO) systems.

    Unlike :class:`PastFutureDataset`, which draws both windows from a single
    time series, this pairs two distinct series per trial: the *past* window is
    taken from an input series and the *future* window from a (possibly
    differently-dimensioned) output series. This is the natural setup for
    capturing the I/O complexity of, e.g., a single neuron whose many-dimensional
    input spike train drives a one-dimensional output spike train.

    Parameters
    ----------
    input_list : list of numpy.ndarray
        Input time series, each of shape (T_i, N_in).
    output_list : list of numpy.ndarray
        Output time series, each of shape (T_i, N_out). Must match ``input_list``
        in length and in per-trial number of time steps T_i (N_out may differ
        from N_in).
    window_size : int
        Length of the past (input) and future (output) windows (in time steps).

    Notes
    -----
    For trial ``k``, ``__getitem__`` returns ``(input_k[i:i+W], output_k[i+W:i+2W])``
    so the model predicts the *next* output window from the *current* input
    window. Windows are built lazily so memory stays O(sum_i T_i * N) rather than
    materializing all pairs.

    Because ``N_out`` may differ from ``N_in``, this dataset is intended for
    ``predictive_space="observation"`` (the future is used raw, never encoded).
    Pass explicit ``critic_params`` to ``CPIC`` (e.g. ``{"x_dim": T*ydim,
    "y_dim": T*N_out, "hidden_dim": ...}``) so the predictive critic/baseline are
    sized for the output dimension rather than the input dimension.
    """

    def __init__(self, input_list, output_list, window_size):
        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if len(input_list) != len(output_list):
            raise ValueError(
                f"input_list and output_list must have the same number of trials; "
                f"got {len(input_list)} and {len(output_list)}"
            )

        self.input_list = [np.asarray(ts) for ts in input_list]
        self.output_list = [np.asarray(ts) for ts in output_list]
        self._segments = []
        cum = 0
        for k, (inp, out) in enumerate(zip(self.input_list, self.output_list)):
            T_in = inp.shape[0]
            T_out = out.shape[0]
            if T_in != T_out:
                raise ValueError(
                    f"Trial {k}: input and output must have the same number of time steps; "
                    f"got T_in={T_in}, T_out={T_out}"
                )
            nw = T_in - 2 * self.window_size
            if nw <= 0:
                raise ValueError(
                    f"Trial {k}: each series needs length > 2*window_size; "
                    f"got T={T_in}, window_size={self.window_size}"
                )
            self._segments.append({"inp": inp, "out": out, "base": cum, "nw": nw})
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
                return seg["inp"][i : i + W], seg["out"][i + W : i + 2 * W]
        raise IndexError(idx)
