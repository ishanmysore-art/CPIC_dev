# particle_experiment config layout

Each config's `[Paths] output_dir` points at the matching `res/<name>` directory.

## Objective

All configs use the soft-penalty predictive-information objective

```
L = beta * I_compress - beta1 * I_predictive
```

with `estimator_compress = infonce_upper` (an UPPER bound on the minimized rate
`I(X;Z)`; a lower bound is degenerate to minimize). `beta` sets where on the
rate/prediction curve the model sits.

## Configs

| config | produces | notes |
|--------|----------|-------|
| `config_particle_beta_sweep.ini` | `res/beta_sweep/` | soft-penalty β sweep mapping the compression/prediction tradeoff (3 encoders × latent/obs). Run on NERSC. |
| `config_particle_noise_sweep.ini` | `res/noise_sweep/` | latent-quality R² families (position/velocity/noise) vs number of noise particles |
| `config_particle_blob_sweep.ini` | `res/blob_sweep/` | same families vs number of coherent-orbit blob particles |
| `config_particle_noise_sweep_noblob.ini` | `res/noise_sweep_noblob/` | velocity-without-blob noise sweep (num_blob=0): is obs-space signal coming from structured noise? |
| `config_particle_nonclosed_noise_sweep.ini` | `res/nonclosed_noise_sweep/` | non-closed (2-torus) trajectory variant so velocity R² and position R² diverge |
| `config_particle_noblob_control.ini` | `res/noblob_control/` | num_blob=0 negative control for the velocity metric |
| `config_particle_probe_figs.ini` | `res/probe_figs/` | per-condition orbit + velocity probe figures |

## Named-config shortcuts

`run_particle_experiment.py` accepts `--config <shortname>` aliases as well as a full
path (`--config experiments/particle_experiment/config/…`). The full-path form is the
normal way to run these configs.
