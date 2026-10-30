import h5py
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import resample
from scipy.ndimage import convolve1d


def compute_R2(X_inf, X_true):
    """
    compute R2 value
    :param X_inf: inferred X dim T x N
    :param X_true: true X dim T x N
    :return:
    """
    return 1 - np.sum((X_inf - X_true)**2) / np.sum(
        (X_true - np.mean(X_true, axis=0)) ** 2)


def linear_alignment(X_inf, X_true):
    """
    Perform linear alignment of inferred data to true data using least-squares regression.

    This function computes the optimal linear transformation (matrix beta) that 
    best maps the inferred data X_inf to the true data X_true in the least-squares sense.
    It returns the transformed X_inf aligned to X_true.

    Parameters
    ----------
    X_inf : np.ndarray
        Inferred data. Shape: (n_samples, n_features) or (T, N).
    X_true : np.ndarray
        True data to align to. Shape: (n_samples, n_targets) or (T, N).

    Returns
    -------
    X_trans : np.ndarray
        Linearly transformed version of X_inf aligned to X_true.
    """
    beta = np.linalg.lstsq(X_inf, X_true, rcond=None)[0]
    X_trans = np.dot(X_inf, beta)
    return X_trans


def sum_over_chunks(X, stride):
    X_trunc = X[:len(X) - (len(X) % stride)]
    reshaped = X_trunc.reshape((len(X_trunc) // stride, stride, X.shape[1]))
    summed = reshaped.sum(axis=1)
    return summed


def moving_center(X, n, axis=0):
    if n % 2 == 0:
        n += 1
    w = -np.ones(n) / n
    w[n // 2] += 1
    X_ctd = convolve1d(X, w, axis=axis)
    return X_ctd


def calc_autocorr_fns(X, T):
    autocorr_fns = np.zeros((X.shape[1], T))
    for dt in range(T):
        autocorr_fns[:, dt] = np.sum((X[dt:] * X[:len(X) - dt]), axis=0) / (len(X) - dt)
    return autocorr_fns


def load_kording_paper_data(filename, bin_width_s=0.05, min_spike_count=10, preprocess=True):
    with open(filename, "rb") as fname:
        data = pickle.load(fname)
    X, Y = data[0], data[1]
    good_X_idx = (1 - (np.isnan(X[:, 0]) + np.isnan(X[:, 1]))).astype(bool)
    good_Y_idx = (1 - (np.isnan(Y[:, 0]) + np.isnan(Y[:, 1]))).astype(bool)
    good_idx = good_X_idx * good_Y_idx
    X, Y = X[good_idx], Y[good_idx]
    chunk_size = int(np.round(bin_width_s / 0.05))  # 50 ms default bin width
    X, Y = sum_over_chunks(X, chunk_size), sum_over_chunks(Y, chunk_size) / chunk_size
    X = X[:, np.sum(X, axis=0) > min_spike_count]
    if preprocess:
        X = np.sqrt(X)
        X = moving_center(X, n=600)
        Y -= Y.mean(axis=0, keepdims=True)
        Y /= Y.std(axis=0, keepdims=True)
    return {'neural': X, 'loc': Y}


def load_weather_data(filename):
    df = pd.read_csv(filename)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df[['Vancouver', 'Portland', 'San Francisco', 'Seattle',
             'Los Angeles', 'San Diego', 'Las Vegas', 'Phoenix', 'Albuquerque',
             'Denver', 'San Antonio', 'Dallas', 'Houston', 'Kansas City',
             'Minneapolis', 'Saint Louis', 'Chicago', 'Nashville', 'Indianapolis',
             'Atlanta', 'Detroit', 'Jacksonville', 'Charlotte', 'Miami',
             'Pittsburgh', 'Toronto', 'Philadelphia', 'New York', 'Montreal',
             'Boston']]
    df = df.dropna(axis=0, how='any')
    dts = (df.index[1:] - df.index[:-1]).to_numpy()
    df = df.iloc[np.nonzero(dts > dts.min())[0].max() + 1:]
    Xfs = df.values.copy()
    ds_factor = 24
    X = resample(Xfs, Xfs.shape[0] // ds_factor, axis=0)
    return X


"""
Download .mat files from
https://zenodo.org/record/583331#.XNtzE5NKjys
Longest session (only has M1): indy_20160627_01.mat
TODO: use downsampling w/ scipy.signal instead of decimation
"""


def load_sabes_data(filename, bin_width_s=.05, preprocess=True):
    # Load MATLAB file
    with h5py.File(filename, "r") as f:
        # Get channel names (e.g. M1 001 or S1 001)
        n_channels = f['chan_names'].shape[1]
        chan_names = []
        for i in range(n_channels):
            chan_names.append(f[f['chan_names'][0, i]][()].tobytes()[::2].decode())
        # Get M1 and S1 indices
        M1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'M1']
        S1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'S1']
        # Get time
        t = f['t'][0, :]
        # Individually process M1 and S1 indices
        result = {}
        for indices in (M1_indices, S1_indices):
            if len(indices) == 0:
                continue
            # Get region (M1 or S1)
            region = chan_names[indices[0]].split(" ")[0]
            # Perform binning
            n_channels = len(indices)
            n_sorted_units = f["spikes"].shape[0] - 1  # The FIRST one is the 'hash' -- ignore!
            d = n_channels * n_sorted_units
            max_t = t[-1]
            n_bins = int(np.floor((max_t - t[0]) / bin_width_s))
            binned_spikes = np.zeros((n_bins, d), dtype=np.int_)
            for chan_idx in indices:
                for unit_idx in range(1, n_sorted_units):  # ignore hash!
                    spike_times = f[f["spikes"][unit_idx, chan_idx]][()]
                    if spike_times.shape == (2,):
                        # ignore this case (no data)
                        continue
                    spike_times = spike_times[0, :]
                    # get rid of extraneous t vals
                    spike_times = spike_times[spike_times - t[0] < n_bins * bin_width_s]
                    bin_idx = np.floor((spike_times - t[0]) / bin_width_s).astype(np.int_)
                    unique_idxs, counts = np.unique(bin_idx, return_counts=True)
                    # make sure to ignore the hash here...
                    binned_spikes[unique_idxs, chan_idx * n_sorted_units + unit_idx - 1] += counts
            binned_spikes = binned_spikes[:, binned_spikes.sum(axis=0) > 0]
            if preprocess:
                binned_spikes = binned_spikes[:, binned_spikes.sum(axis=0) > 5000]
                binned_spikes = np.sqrt(binned_spikes)
                binned_spikes = moving_center(binned_spikes, n=600)
            result[region] = binned_spikes
        # Get cursor position
        cursor_pos = f["cursor_pos"][:].T
        # Line up the binned spikes with the cursor data
        t_mid_bin = np.arange(len(binned_spikes)) * bin_width_s + bin_width_s / 2
        cursor_pos_interp = interp1d(t - t[0], cursor_pos, axis=0)
        cursor_interp = cursor_pos_interp(t_mid_bin)
        if preprocess:
            cursor_interp -= cursor_interp.mean(axis=0, keepdims=True)
            cursor_interp /= cursor_interp.std(axis=0, keepdims=True)
        result["cursor"] = cursor_interp
        return result


def _read_sabes_chan_names(f):
    return [f[f['chan_names'][0, i]][()].tobytes()[::2].decode()
            for i in range(f['chan_names'].shape[1])]


def _read_sabes_spike_times(f, unit_row, chan_idx):
    """Spike times for one cell of the MATLAB ``spikes`` array, or None if the cell is empty."""
    spike_times = f[f["spikes"][unit_row, chan_idx]][()]
    if spike_times.shape == (2,):  # MATLAB stores an empty cell as a length-2 placeholder
        return None
    return spike_times.reshape(-1)


def load_sabes_session(filename, bin_width_s=.05, region="M1", representation="unit",
                       legacy_indexing=False):
    """
    Bin one Sabes session into raw spike counts while keeping column identities.

    Unlike ``load_sabes_data``, no filtering, square-root, drift removal or behavioral
    standardization is applied here: every learned or data-dependent statistic must be
    fit on a training split (see ``cpic.exp_utils.sessions``).

    Parameters
    ----------
    filename : str
        Path to a Sabes ``.mat`` (v7.3 / HDF5) session file.
    bin_width_s : float
        Bin width in seconds.
    region : str
        Channel-name prefix to keep ("M1" or "S1").
    representation : str
        "unit": one column per non-empty sorted-unit cell (rows 1..u-1 of ``spikes``; row 0
        is the unsorted hash). Columns have no cross-session correspondence.
        "electrode": one column per channel of ``region``, summing every row including hash,
        i.e. all threshold crossings. Columns correspond to physical electrodes, not neurons.
    legacy_indexing : bool
        With representation="unit", reproduce ``load_sabes_data``'s unit selection, which
        visits rows 1..u-2 and never the last sorted-unit row.

    Returns
    -------
    dict with
        counts : (n_bins, n_cols) int64 spike counts
        channel : (n_cols,) global channel index of each column
        unit : (n_cols,) row of ``spikes`` (-1 for electrode columns)
        chan_names : list of all channel names in the file
        t_mid : (n_bins,) bin centers in seconds relative to the first sample
        cursor_pos, target_pos, finger_pos : kinematics interpolated at bin centers, raw units
        bin_width_s, duration_s, n_spikes_outside, n_spike_rows : bookkeeping
    """
    if representation not in ("unit", "electrode"):
        raise ValueError(f"Unknown representation: {representation!r}")
    if legacy_indexing and representation != "unit":
        raise ValueError("legacy_indexing only applies to representation='unit'")
    with h5py.File(filename, "r") as f:
        chan_names = _read_sabes_chan_names(f)
        chan_indices = [i for i, name in enumerate(chan_names) if name.split(" ")[0] == region]
        if len(chan_indices) == 0:
            raise ValueError(f"No {region} channels in {filename}")
        t = f['t'][0, :]
        n_rows = f["spikes"].shape[0]
        n_bins = int(np.floor((t[-1] - t[0]) / bin_width_s))
        if representation == "electrode":
            unit_rows = range(0, n_rows)
        elif legacy_indexing:
            unit_rows = range(1, n_rows - 1)
        else:
            unit_rows = range(1, n_rows)

        columns, channel, unit = [], [], []
        n_spikes_outside = 0
        for chan_idx in chan_indices:
            electrode_counts = np.zeros(n_bins, dtype=np.int64)
            for unit_row in unit_rows:
                spike_times = _read_sabes_spike_times(f, unit_row, chan_idx)
                if spike_times is None:
                    continue
                rel = spike_times - t[0]
                inside = (rel >= 0) & (rel < n_bins * bin_width_s)
                n_spikes_outside += int((~inside).sum())
                bin_counts = np.bincount(np.floor(rel[inside] / bin_width_s).astype(np.int64),
                                         minlength=n_bins)
                if representation == "electrode":
                    electrode_counts += bin_counts
                else:
                    columns.append(bin_counts)
                    channel.append(chan_idx)
                    unit.append(unit_row)
            if representation == "electrode":
                columns.append(electrode_counts)
                channel.append(chan_idx)
                unit.append(-1)

        t_mid = np.arange(n_bins) * bin_width_s + bin_width_s / 2
        kinematics = {}
        for key in ("cursor_pos", "target_pos", "finger_pos"):
            if key in f:
                kinematics[key] = interp1d(t - t[0], f[key][:].T, axis=0)(t_mid)

    counts = (np.stack(columns, axis=1) if columns
              else np.zeros((n_bins, 0), dtype=np.int64))
    return {
        "counts": counts,
        "channel": np.asarray(channel, dtype=np.int64),
        "unit": np.asarray(unit, dtype=np.int64),
        "chan_names": chan_names,
        "t_mid": t_mid,
        "bin_width_s": float(bin_width_s),
        "duration_s": float(t[-1] - t[0]),
        "n_spikes_outside": n_spikes_outside,
        "n_spike_rows": int(n_rows),
        "representation": representation,
        "legacy_indexing": bool(legacy_indexing),
        **kinematics,
    }


def load_accel_data(filename, preprocess=True):
    df = pd.read_csv(filename)
    X = df.values[:, 1:]
    if preprocess:
        X -= X.mean(axis=0, keepdims=True)
        X /= X.std(axis=0, keepdims=True)
    return X


class CrossValidate:
    def __init__(self, X, Y, num_folds, stack=True):
        self.X, self.Y = X, Y
        self.num_folds = num_folds
        self.idxs = np.array_split(np.arange(len(X)), num_folds)
        self.stack = stack

    def __iter__(self):
        self.fold_idx = 0
        return self

    def __next__(self):
        fold_idx = self.fold_idx
        if fold_idx == self.num_folds:
            raise StopIteration

        test_idxs = self.idxs[fold_idx]
        train_idxs = []
        if fold_idx > 0:
            train_idxs.append(np.concatenate([self.idxs[ii] for ii in range(fold_idx)]))
        if fold_idx < self.num_folds - 1:
            train_idxs.append(np.concatenate([self.idxs[ii]
                                              for ii in range(fold_idx + 1, self.num_folds)]))

        X, Y = self.X, self.Y
        X_test = X[test_idxs]
        Y_test = Y[test_idxs]
        if self.stack:
            X_train = np.concatenate([X[idxs] for idxs in train_idxs])
            Y_train = np.concatenate([Y[idxs] for idxs in train_idxs])
        else:
            X_train = [X[idxs] for idxs in train_idxs]
            Y_train = [Y[idxs] for idxs in train_idxs]

        self.fold_idx += 1
        return X_train, X_test, Y_train, Y_test, fold_idx
    