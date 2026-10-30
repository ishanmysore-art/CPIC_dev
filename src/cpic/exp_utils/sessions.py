"""
Chronological splitting and train-only preprocessing for single recording sessions.

Every data-dependent statistic (unit selection, centering means) is fit on the
training segment only, and drift removal is causal and restarted inside each
segment, so no value from validation or test bins can influence processed training
data (or vice versa). Behavior is never touched here.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Splits:
    """Half-open [start, stop) bin ranges for chronological train / val / test segments."""
    train: tuple
    val: tuple
    test: tuple
    gap: int

    def segment(self, name):
        start, stop = getattr(self, name)
        return slice(start, stop)

    def as_dict(self):
        return {"train": list(self.train), "val": list(self.val),
                "test": list(self.test), "gap": self.gap}


def chronological_splits(n_bins, fractions=(0.7, 0.1, 0.2), gap=30):
    """
    Split ``n_bins`` time bins into consecutive train / val / test segments.

    ``gap`` bins are discarded between consecutive segments. It should be at least
    2*T + decoding window + max decoding offset so that no CPIC window or lagged
    decoder sample straddles two segments.
    """
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must be three values summing to 1; got {fractions}")
    usable = n_bins - 2 * gap
    if usable <= 0:
        raise ValueError(f"n_bins={n_bins} is too small for gap={gap}")
    n_train = int(np.floor(usable * fractions[0]))
    n_val = int(np.floor(usable * fractions[1]))
    train = (0, n_train)
    val = (n_train + gap, n_train + gap + n_val)
    test = (val[1] + gap, n_bins)
    return Splits(train=train, val=val, test=test, gap=gap)


def causal_moving_center(X, n):
    """
    Subtract a trailing mean over the last ``n`` bins (including the current bin).

    The window shrinks at the start of ``X`` instead of reaching outside it, so the
    output at bin t depends only on X[max(0, t - n + 1) : t + 1].
    """
    X = np.asarray(X, dtype=np.float64)
    csum = np.cumsum(X, axis=0)
    trailing = csum.copy()
    trailing[n:] = csum[n:] - csum[:-n]
    window = np.minimum(np.arange(1, len(X) + 1), n)[:, None]
    return X - trailing / window


@dataclass
class SessionPreprocessor:
    """
    Train-fit neural preprocessing: rate filter -> sqrt -> causal drift removal -> centering.

    min_rate_hz=None disables the rate filter (e.g. to keep a fixed electrode coordinate
    system across sessions). ``keep`` may be passed to ``fit`` to impose an externally
    chosen column set, e.g. the legacy 109-unit set, without refitting it.
    """
    bin_width_s: float = 0.05
    min_rate_hz: float = 0.5
    sqrt: bool = True
    drift_window_s: float = 30.0
    center: bool = True
    keep: np.ndarray = field(default=None, repr=False)
    mean: np.ndarray = field(default=None, repr=False)

    @property
    def drift_window_bins(self):
        if not self.drift_window_s:
            return 0
        return int(round(self.drift_window_s / self.bin_width_s))

    def _transform_unfitted(self, counts):
        X = np.asarray(counts, dtype=np.float64)[:, self.keep]
        if self.sqrt:
            X = np.sqrt(X)
        if self.drift_window_bins:
            X = causal_moving_center(X, self.drift_window_bins)
        return X

    def fit(self, train_counts, keep=None):
        train_counts = np.asarray(train_counts)
        if keep is not None:
            keep = np.asarray(keep, dtype=bool)
            if keep.shape != (train_counts.shape[1],):
                raise ValueError("keep must have one entry per column")
        elif self.min_rate_hz is None:
            keep = np.ones(train_counts.shape[1], dtype=bool)
        else:
            rates = train_counts.sum(axis=0) / (len(train_counts) * self.bin_width_s)
            keep = rates >= self.min_rate_hz
        self.keep = keep
        X = self._transform_unfitted(train_counts)
        self.mean = X.mean(axis=0) if self.center else np.zeros(X.shape[1])
        return self

    def transform(self, segment_counts, dtype=np.float32):
        """Transform one contiguous segment; drift removal never crosses its boundaries."""
        if self.keep is None:
            raise RuntimeError("SessionPreprocessor must be fit before transform")
        return (self._transform_unfitted(segment_counts) - self.mean).astype(dtype)

    def state(self):
        return {"bin_width_s": self.bin_width_s, "min_rate_hz": self.min_rate_hz,
                "sqrt": self.sqrt, "drift_window_s": self.drift_window_s,
                "center": self.center, "keep": self.keep, "mean": self.mean}


def legacy_unit_mask(counts, count_threshold=5000):
    """
    The legacy ``load_sabes_data`` unit selection: total spikes over the WHOLE session.

    This uses test bins and is length-dependent; it exists only to hold the unit set
    fixed while attributing other corrections (ladder rung L3a).
    """
    return np.asarray(counts).sum(axis=0) > count_threshold


def prepare_session(session, splits, preprocessor, keep=None):
    """
    Split a session from ``load_sabes_session`` and apply a train-fit preprocessor.

    Returns {"X": {split: (n, d) float32}, "Y": {split: {kinematic: array}}, "columns": ...}.
    Kinematics are sliced to the same bins but not transformed.
    """
    counts = session["counts"]
    preprocessor.fit(counts[splits.segment("train")], keep=keep)
    X = {name: preprocessor.transform(counts[splits.segment(name)])
         for name in ("train", "val", "test")}
    kinematic_keys = [k for k in ("cursor_pos", "target_pos", "finger_pos") if k in session]
    Y = {name: {k: session[k][splits.segment(name)] for k in kinematic_keys}
         for name in ("train", "val", "test")}
    columns = {"channel": session["channel"][preprocessor.keep],
               "unit": session["unit"][preprocessor.keep]}
    return {"X": X, "Y": Y, "columns": columns, "splits": splits,
            "preprocessor": preprocessor.state()}
