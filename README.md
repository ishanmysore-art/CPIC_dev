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

## Repository Structure

### Core Package
- CPIC implementation: <code>src/cpic/CPIC.py</code>
- Encoders, critics, and baselines: <code>src/cpic/models.py</code>
- Mutual information estimation: <code>src/cpic/mi.py</code>
- Utility modules: <code>src/cpic/utils/</code>

### Experiments
- Lorenz synthetic experiment: <code>experiments/synthetic_lorenz_experiment/</code>
  - Data generation: <code>synthetic_generator.py</code>, <code>data_generator.py</code>
  - CPIC training/evaluation: <code>synthetic_experiment.py</code>, <code>synthetic_sparse_experiment.py</code>
  - Competitor baselines: <code>synthetic_competitors.py</code>
  - Post-analysis and plotting: <code>synthetic_visualization.py</code>, <code>synthetic_summarization.py</code>, <code>PI_analysis.py</code>, <code>PI_plot.py</code>
- Real-data experiments: <code>experiments/real_data_experiments/</code>
  - CPIC experiments (M1, HC, Temp, MS): <code>real_data_experiment_standard.py</code>, <code>real_data_experiment_standard_beta.py</code>
  - Competitor baselines: <code>real_data_competitors.py</code>
  - Summaries/post-analysis: <code>real_data_summary_standard.py</code>, <code>real_data_summary_standard_beta.py</code>
- Particle dynamics experiment: <code>experiments/particle_experiment/</code>
  - Data generation script: <code>generate_particle_dynamics.py</code> (circle / ellipse / torus trajectories)
  - Config-driven batch encoder sweep (β sweep, held-out velocity/noise probes, num_blob sweep): <code>run_particle_experiment.py</code>
  - Latent-quality metric plots (position vs velocity vs noise-cloud R² + predictive info): <code>plot_latent_metrics.py</code>
  - Complexity curve (predictivity vs compression complexity I(X;Z) over the β sweep): <code>plot_complexity_curve.py</code>
  - Convergence curves: <code>plot_convergence_curves.py</code>; re-plot saved CSV results: <code>plot_particle_results.py</code>
  - Probe-result figures (orbit overlay + velocity field): <code>probe_visualization.py</code>
  - Notebook workflows: <code>particle_dynamics.ipynb</code>, <code>conv_physical_analysis.ipynb</code>
- Video experiment: <code>experiments/video_experiment/</code>
  - Sparse CPIC training/evaluation: <code>run_sparse_cpic.py</code>
  - Train then visualize with one matching <code>--seed</code> / <code>--signature</code>: <code>run_sparse_cpic_train_and_visualize.py</code> (see README, “Train and visualize in one step”)
  - Offline plots from saved checkpoints and pickles: <code>visualize_sparse_cpic_outputs.py</code> (see README, “Visualizing sparse CPIC outputs”)
  - Dryad Chicago Motion download helper: <code>download_dryad_dataset.py</code> (see README, “Natural movie stimulus data”)

## Encoder types
CPIC supports multiple encoder architectures via <code>encoder_params["encoder_type"]</code>. All encoders map input (T x D) to output (T x M). Unless specified otherwise, <code>deterministic</code> defaults to <code>False</code>. Example configurations:

- **Linear**: <code>{"encoder_type": "linear"}</code>
- **MLP**: <code>{"encoder_type": "mlp", "n_layers": 1, "activation": "relu"}</code>
- **2x MLP**: <code>{"encoder_type": "mlp2"}</code>
- **Conv (spatial)**: <code>{"encoder_type": "conv_spatial"}</code>, with optional <code>n_layers</code>, <code>conv_kernel_size</code>, <code>conv_stride</code>, <code>conv_padding</code> — 2D conv over the feature axis
- **Conv (particle)**: <code>{"encoder_type": "conv_particle", "conv_kernel_size": 5}</code>, with optional <code>conv_stride</code>, <code>conv_padding</code>, <code>conv_coord_kernel_size</code>, <code>n_layers</code> — 2D conv over (particles x coords)
- **Conv (physical / occupancy grid)**: <code>{"encoder_type": "conv_physical", "grid_size": 20, "spatial_bounds": 3.0}</code>, with optional <code>conv_kernel_size</code>, <code>conv_stride</code>, <code>conv_padding</code>, <code>n_layers</code> — bins particle positions into a spatial grid, then a shared Conv2d
- **Conv (spatiotemporal)**: <code>{"encoder_type": "conv_spatiotemporal", "conv_kernel_size": 3, "conv_stride": 1}</code>, with optional <code>conv_padding</code>, <code>n_layers</code> — 2D conv over both feature and time axes
- **1D temporal Conv**: <code>{"encoder_type": "conv_temporal", "kernel_size_1d": 3, "n_layers": 1}</code>
- **Feature-mask MLP**: <code>{"encoder_type": "mask_mlp", "mask_learnable": true, "mask_init": "uniform"}</code>, with optional <code>mask_init_values</code>, <code>n_layers</code>, <code>activation</code> — per-feature Bernoulli straight-through gate before an MLP

Pass <code>encoder_params</code> when constructing CPIC, and when calling <code>fit()</code> the encoder is built from the registry.

## Configuration
Experiment configurations are grouped under:
- <code>experiments/synthetic_lorenz_experiment/config/</code>
- <code>experiments/real_data_experiments/config/</code>
- <code>experiments/video_experiment/config/</code>
- <code>experiments/particle_experiment/config/</code>

## Figures
Figures and generated outputs are stored inside each experiment folder, e.g.:
- <code>experiments/synthetic_lorenz_experiment/fig/</code>
- <code>experiments/real_data_experiments/fig/</code>

## Natural movie stimulus data (Dryad)

Source: [Dryad dataset](https://doi.org/10.5061/dryad.4qrfj6qm8) — stimulus and recordings from the Chicago Motion Database, as used in *Stimulus-invariant aspects of the retinal code drive discriminability of natural scenes* (2024). If you use this data in a publication, cite that paper and the Dryad record.

### Downloading from Dryad (`experiments/video_experiment/download_dryad_dataset.py`)

This script uses Dryad’s HTTP API to download the published files (AVIs, MATLAB archives, and Dryad’s dataset `README.md`) into `data/dryad_chicago_natural_movies/` at the repository root by default. The `data/` tree is gitignored.

Dryad’s file endpoints require OAuth **client credentials** (not anonymous downloads). After you create a Dryad account with ORCID and add an API application under *My account*, set:

```bash
export DRYAD_CLIENT_ID="..."
export DRYAD_CLIENT_SECRET="..."
```

See [Dryad API accounts](https://github.com/datadryad/dryad-app/blob/main/documentation/apis/api_accounts.md). The script’s module docstring summarizes the same steps.

List remote files without credentials or downloading:

```bash
uv run python experiments/video_experiment/download_dryad_dataset.py --dry-run
```

Download all files (requires the environment variables above):

```bash
uv run python experiments/video_experiment/download_dryad_dataset.py
```

Use `--output-dir PATH` for a different destination, or `uv run python experiments/video_experiment/download_dryad_dataset.py --help` for flags such as `--force` and `--domain` / `DRYAD_OAUTH_DOMAIN`.

If you prefer not to use the script, unpack a manually obtained copy to a directory on your machine. The default `--input` in `scripts/process_avi_to_numpy.py` is `data/dryad_chicago_natural_movies/MultipleMoviesStim_1_tree.avi` under this repository’s root (the same layout as `download_dryad_dataset.py`); override `--input` if your AVI lives elsewhere.

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
  --input data/dryad_chicago_natural_movies/MultipleMoviesStim_4_fish.avi
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

### Training sparse CPIC on video (`experiments/video_experiment`)

Point `[User]` `video_path` in `experiments/video_experiment/config/config_video_sparse_cpic.ini` at your `.npz` (or `.npy`) frames, and adjust `saved_root`, `device`, and hyperparameters as needed.

From `experiments/video_experiment/`:

```bash
cd experiments/video_experiment
python run_sparse_cpic.py --config config/config_video_sparse_cpic.ini
```

If you use `uv` from the repo root without activating a venv, the same invocation is:

```bash
cd experiments/video_experiment
uv run python run_sparse_cpic.py --config config/config_video_sparse_cpic.ini
```

From the repository root (equivalent):

```bash
uv run python experiments/video_experiment/run_sparse_cpic.py \
  --config experiments/video_experiment/config/config_video_sparse_cpic.ini
```

Optional CLI flags include `--seed`, `--signature`, and `--device` (see the script’s `--help`). Defaults: `--seed` is `22`; `--signature` is the local wall-clock time as an integer `YYYYMMDDHHMMSS`, and names the run subfolder under `tensor_logs` and the checkpoint filename (below).

### Train and visualize in one step (`run_sparse_cpic_train_and_visualize.py`)

This wrapper runs `run_sparse_cpic.py` and then `visualize_sparse_cpic_outputs.py` with the **same** `--seed` and `--signature`, so visualization paths line up with the artifacts training just wrote. If you omit `--signature`, a single timestamp is chosen at startup and passed to both steps (same behavior as relying on each script’s default).

From the repository root:

```bash
uv run python experiments/video_experiment/run_sparse_cpic_train_and_visualize.py
```

Explicit run id and seed:

```bash
uv run python experiments/video_experiment/run_sparse_cpic_train_and_visualize.py \
  --config experiments/video_experiment/config/config_video_sparse_cpic.ini \
  --seed 42 \
  --signature 20260513120000
```

Optional: `--device` (training only), `--frame-shape H W` (visualization only), `--saved-root` (visualization only; default is `User.saved_root` from the config and should match where training wrote). See the script’s `--help`.

### Visualizing sparse CPIC outputs (`experiments/video_experiment/visualize_sparse_cpic_outputs.py`)

After a run, this script reads the saved artifacts under `saved_root` and writes PNG figures (and a reconstructed preview GIF when frame shape matches the model) into:

`<saved_root>/visualizations_sig<signature>_seed<seed>/`

It expects:

- `encoded_representations_seed<seed>.pkl`
- `sparse_cpic_checkpoint_sig<signature>_seed<seed>.pt` (loads `decoder.weight`)
- `inferred_trials_seed<seed>.pkl` (reconstructed trial used for frame samples and GIF)

Typical outputs include encoded-representation heatmaps and PC trajectory plots, decoder weight matrix and column norms (plus per-latent RGB basis tiles when dimensions match `H×W×3`), and reconstructed frame samples plus `reconstructed_video.gif` when GIF writing succeeds.

From the repository root (use the **same** `--seed` and `--signature` you used for training; `saved_root` must match `[User]` `saved_root` in the config unless you moved files):

```bash
uv run python experiments/video_experiment/visualize_sparse_cpic_outputs.py \
  --saved-root res/video_sparse_cpic \
  --seed 22 \
  --signature 20260513120000
```

Optional: `--config PATH` (defaults to `experiments/video_experiment/config/config_video_sparse_cpic.ini`) or `--frame-shape H W`. Requires a working Matplotlib install (use `uv run` from the project environment). See the script’s `--help` for full flags.

### Visualizing `tensor_logs` (TensorBoard)

Training uses [TensorBoardX](https://github.com/lanpa/tensorboardX) and writes event files under:

`<saved_root>/tensor_logs/<signature>/`

where `saved_root` comes from the config `[User]` section and `signature` from `--signature` (default: `YYYYMMDDHHMMSS` wall time when you start training). With the example `saved_root = res/video_sparse_cpic`, that path usually resolves to the repository root (see `run_sparse_cpic.py` if you also keep a copy under `experiments/video_experiment/res/`).

Install is already covered by the main dependencies (`tensorboard` and `tensorboardX` in `pyproject.toml`). From the repository root, point TensorBoard at the `tensor_logs` parent so you can compare multiple signatures in one UI:

```bash
uv run tensorboard --logdir res/video_sparse_cpic/tensor_logs
```

Then open the URL TensorBoard prints (by default `http://localhost:6006/`). To view a single run only (replace the folder name with your run’s `signature`):

```bash
uv run tensorboard --logdir res/video_sparse_cpic/tensor_logs/20260513120000
```

Replace `res/video_sparse_cpic` with your `saved_root` if you changed it in the INI.