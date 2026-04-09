# Compressed Predictive Information Coding 
This repo is used to publish the code for Compressed Predictive Information Coding (CPIC) methodology.

## Installation

### Using uv (recommended)

The project is set up for [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
# Create a virtual environment and install the package and dependencies (editable)
uv sync

# Include dev dependencies (e.g. pytest)
uv sync --dev

# Include the optional DCA extra (DynamicalComponentsAnalysis, NumPy<2)
uv sync --extra dca
```

Run code with:
```bash
uv run python your_script.py
uv run pytest
```

A `.python-version` file (3.10) and `uv.lock` are included for consistent installs.

### Using pip

Install CPIC with:
```bash
pip install cpic
```

For **DCA initialization** (e.g. `DCA_init`, `load_sabes_data`): install the optional dependency:
```bash
pip install "cpic[dca]"
```
(Quotes are required in zsh so `[dca]` is not interpreted as a glob.) This installs DynamicalComponentsAnalysis and pins NumPy<2 for compatibility. See https://dynamicalcomponentsanalysis.readthedocs.io/en/latest/index.html.

## Code

### Main Code
1. Main code:
   1. CPIC code: <code>src/cpic/CPIC.py</code>
   2. Encoders, Critics, Baselines: <code>src/cpic/models.py</code>
   3. Mutual information estimation: <code>src/cpic/mi.py</code>
   4. utility code: <code>src/cpic/utils/</code>

### Synthetic Experiments

1. Lorenz experiment code: <code>synthetic/synthetic_*.py</code>
   1. Synthetic data generation using <code>synthetic_generator.py</code>.
   2. CPIC model on synthetic data using <code>synthetic_experiment.py</code>.
   3. Other models on synthetic data using <code>synthetic_competitors.py</code>.
   4. Per/Post analysis includes <code>synthetic_visualization.py</code> and <code>synthetic_summarization.py</code>.
2. Synthetic experiments to understand the Prediction information in CPIC setting:
   1. Synthetic data generation with <code>synthetic/data_generator.py</code>.
   2. PI analysis using <code>synthetic/PI_analysis.py</code>.


### Real Data Experiments
1. Real data experiments code: <code>analysis/real_data_*.py</code>
   1. Real data experiments with CPIC for four datasets including M1, HC, Temp, MS using <code>real_data_experiment_standard.py</code>, <code>real_data_experiment_standard_beta.py</code>. Note that beta refers to varying weight option.
   2. Real data experiments with other models using <code>real_data_competitors.py</code>.
   3. Real data experiments post analysis: <code>real_data_summary_standard.py</code>, <code>real_data_summary_standard_beta.py</code>.

## Encoder types
CPIC supports multiple encoder architectures via <code>encoder_params["encoder_type"]</code>. All encoders map input (T x D) to output (T x M). Unless specified otherwise, <code>deterministic</code> defaults to <code>False</code>. Example configurations:

- **Linear**: <code>{"encoder_type": "linear"}</code>
- **MLP**: <code>{"encoder_type": "mlp", "n_layers": 1, "activation": "relu"}</code>
- **2x MLP**: <code>{"encoder_type": "mlp2"}</code>
- **Conv (spatial only)**: <code>{"encoder_type": "conv_spatial"}</code> or <code>"conv"</code>, with optional <code>n_layers</code>, <code>conv_kernel_size</code>, <code>conv_stride</code>, <code>conv_padding</code>
- **Conv (spatiotemporal)**: <code>{"encoder_type": "conv_spatiotemporal", "conv_kernel_size": 3, "conv_stride": 1}</code>
- **1D temporal Conv**: <code>{"encoder_type": "conv1d_temporal", "kernel_size_1d": 3, "n_layers": 1}</code>

Pass <code>encoder_params</code> when constructing CPIC, and when calling <code>fit()</code> the encoder is built from the registry.

## Configuration
Synthetic experiment configurations for CPIC are available in <code>synthetic/config/\*</code>. Real data experiment configurations for CPIC are available in <code>analysis/config/\*</code>.

## Figures
Figures for synthetic experiments in the paper are available in <code>synthetic/fig/\*</code>. Figures for real data experiments in the paper are available in <code>analysis/fig/\*</code>.

## Natural movie stimulus data (Dryad)

Source: [Dryad dataset](https://doi.org/10.5061/dryad.4qrfj6qm8) — stimulus and recordings from the Chicago Motion Database, as used in *Stimulus-invariant aspects of the retinal code drive discriminability of natural scenes* (2024). If you use this data in a publication, cite that paper and the Dryad record.

Unpack the archive to a directory on your machine (below we use `/Users/ruimeng/data` as an example). The default `--input` in `scripts/process_avi_to_numpy.py` points at that layout; override `--input` if your path differs.

### Files in the dataset directory

**Movie stimuli (AVI), shown to the retina:**

| File | Content (short) |
|------|------------------|
| `MultipleMoviesStim_1_tree.avi` | Tree in wind |
| `MultipleMoviesStim_2_water.avi` | Water in a small canal |
| `MultipleMoviesStim_3_grasses.avi` | Tall grasses in wind |
| `MultipleMoviesStim_4_fish.avi` | Fish in a tank with plants |
| `MultipleMoviesStim_5_opticflow.avi` | Woods, camera moving through underbrush |

**MATLAB archives (neural data and checkerboard mapping):**

- `movieBinnedSpiking.mat` — binned binary spikes to the movie stimuli (`movnames`, `ncell`, `nmov`, `nreps`, `samplingfreq`, `binned`, …). Responses for movie *i*: `binned(1:nreps(i), :, :, i)`; neuron *j* for movie *i*: `binned(1:nreps(i), :, j, i)`.
- `binaryCheckerboard.mat` — checkerboard RF mapping (`samplingFreq`, `binaryCheckerboard`, `stimulusFrames`).

### Converting AVI stimuli to NumPy (`scripts/process_avi_to_numpy.py`)

The script decodes an AVI to a frame array (and optional preview image). Its `--help` epilog points at this repository’s `README.md` (resolved from `scripts/process_avi_to_numpy.py`, not from your shell’s current directory).

Install the optional video dependency:

```bash
uv sync --extra video
```

Basic run (uses the default `--input` if that file exists):

```bash
uv run python scripts/process_avi_to_numpy.py
```

Explicit paths:

```bash
uv run python scripts/process_avi_to_numpy.py \
  --input /Users/ruimeng/data/MultipleMoviesStim_4_fish.avi
```

### Outputs

- **Default:** writes `<video_stem>.npz` next to the video (unless `--output` is set).
- **`npz` format (default):** `frames` (uint8 `(T, H, W, 3)`, RGB), `fps` (`float32` scalar), `shape` (int64). Load with `numpy.load(..., allow_pickle=False)`.
- **`npy` format:** `--format npy` saves only the frame array; FPS is not stored.
- **Preview PNG:** default `<output_stem>_preview.png`; `--no-preview`, `--show`, `--preview PATH`, `--grid N`.
- **Subsampling:** `--max-frames K`, `--stride N`.

Full CLI: `uv run python scripts/process_avi_to_numpy.py --help`.

### Loading a saved `.npz` in Python

```python
import numpy as np

data = np.load("MultipleMoviesStim_1_tree.npz", allow_pickle=False)
frames = data["frames"]  # (T, H, W, 3), uint8, RGB
fps = float(data["fps"].item())
```

Use the path where you wrote the file if it is not the current working directory.