"""
Per-electrode training-split firing rates for each session (neural-only, no behavior).

Used as the "firing-rate similarity" control predictor in the cross-session analysis:
does the frozen critic's zero-shot bound predict transfer better than simply asking how
similar the electrodes' firing rates are between the two sessions? The training split is
the same chronological 70/10/20 split with gaps used by transfer_sessions.py.

    python experiments/cross_session/electrode_rates.py $SCRATCH/cpic-data/real_data/M1/indy_*.mat --out inventory
"""

import argparse
import json
from pathlib import Path

from cpic.exp_utils.data_util import load_sabes_session
from cpic.exp_utils.sessions import chronological_splits
from transfer_sessions import BIN_WIDTH_S, SPLIT


def electrode_rates(path):
    session = load_sabes_session(str(path), bin_width_s=BIN_WIDTH_S, representation="electrode")
    splits = chronological_splits(len(session["counts"]), **SPLIT)
    train = session["counts"][splits.segment("train")]
    rates = train.sum(axis=0) / (len(train) * BIN_WIDTH_S)
    return {"session": Path(path).stem,
            "chan_names": [session["chan_names"][c] for c in session["channel"]],
            "train_rate_hz": rates.tolist(), "duration_s": session["duration_s"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", default="inventory")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in args.paths:
        stats = electrode_rates(path)
        target = out / f"electrode_rates_{stats['session']}.json"
        target.write_text(json.dumps(stats))
        print(f"{stats['session']}: {len(stats['train_rate_hz'])} electrodes, "
              f"{stats['duration_s'] / 60:.1f} min -> {target}")


if __name__ == "__main__":
    main()
