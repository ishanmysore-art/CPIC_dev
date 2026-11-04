"""
Population-specific read-ins trained against a shared CPIC critic.

The linear CPIC encoder is a per-bin read-in z_t = W x_t; all temporal structure is
scored by the critic. A new neural population (another session, or a disjoint subset of
units) therefore needs its own read-in, and the only object that can carry anything
over from a source population is the critic. ``fit_readin`` trains a new read-in under
one of three matched conditions:

    critic="pretrained"  the source model's critic, frozen        (arm i)
    critic="random"      a freshly initialized critic, frozen     (arm ii: matched control)
    critic="fresh"       a fresh critic trained jointly           (arm iii: no pretraining)

All three use the same read-in initialization, data, optimizer, number of steps and
objective (beta * I_compress - I_predictive), and no behavior.
"""

import copy
import random

import numpy as np
import torch

from .CPIC import CPIC
from .utils.data import PastFutureDataset
from .utils.helpers import DCA_init

# Frozen, pre-registered benchmark settings (config_m1_scaling_benchmark.ini).
BENCHMARK = {"ydim": 32, "T": 4, "beta": 1e-3, "hidden_dim": 256, "batch_size": 512,
             "lr": 1e-3, "epochs": 10, "n_init_dca": 5}
CRITIC_ARMS = ("pretrained", "random", "fresh")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_cpic(xdim, device, cfg=BENCHMARK):
    """Linear stochastic CPIC with the benchmark objective (same kwargs as the legacy script)."""
    T, ydim, hidden = cfg["T"], cfg["ydim"], cfg["hidden_dim"]
    return CPIC(
        ydim=ydim, xdim=xdim, T=T,
        mi_params={"estimator_compress": "infonce_upper", "estimator_predictive": "infonce_lower",
                   "critic": "concat", "baseline": "constant"},
        baseline_params={"hidden_dim": hidden},
        encoder_params={"deterministic": False, "n_layers": 1, "encoder_type": "linear"},
        hidden_dim=hidden, beta=cfg["beta"], beta1=1.0, beta2=0.0, device=device,
        predictive_space="latent", predictive_loss="mi",
        critic_params={"x_dim": T * ydim, "y_dim": T * ydim, "hidden_dim": hidden},
    ).to(device)


def fit_standard(X_train, device, seed, cfg=BENCHMARK, use_dca=True, verbose=False,
                 neuron_dropout_p=0.0):
    """
    The standard single-population CPIC recipe (DCA init, CPIC.fit, fixed epochs).

    ``neuron_dropout_p`` > 0 zero-fills a random subset of input columns per minibatch
    (whole neurons/electrodes, shared by past and future windows), seeded by ``seed``.
    """
    seed_everything(seed)
    init_weights = (DCA_init(X_train, T=cfg["T"], d=cfg["ydim"], n_init=cfg["n_init_dca"])
                    if use_dca else None)
    model = make_cpic(X_train.shape[1], device, cfg)
    model.fit(PastFutureDataset([X_train], window_size=cfg["T"]), init_weights=init_weights,
              epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
              early_stop=cfg["epochs"] + 1, verbose=verbose,
              neuron_dropout_p=neuron_dropout_p, neuron_dropout_seed=seed)
    return model


def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def random_critic_like(critic, seed):
    """A copy of ``critic`` with every layer re-initialized from ``seed`` (global RNG untouched)."""
    # Re-initialize on the CPU so only the (forked) CPU generator is used; the caller moves it.
    fresh = copy.deepcopy(critic).cpu()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for module in fresh.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    return fresh


def transformed_encoder_state(encoder, rotation=None, permutation=None):
    """
    State dict of a linear ``StructuredEncoder`` with its input columns reassigned and/or
    its latent axes rotated, for initializing a read-in near (or deliberately away from)
    the source coordinates.

    permutation : column j of the new read-in gets the source weights of column
        permutation[j] (e.g. a wrong electrode correspondence). Applied to both heads.
    rotation : d x d matrix G; the mean head becomes G W (bias G b). The log-variance head
        is left in its original axes.
    """
    state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    if permutation is not None:
        index = torch.as_tensor(np.asarray(permutation), dtype=torch.long)
        for key in ("_mean.weight", "_logvars.weight"):
            state[key] = state[key][:, index.to(state[key].device)]
    if rotation is not None:
        G = torch.as_tensor(np.asarray(rotation), dtype=state["_mean.weight"].dtype,
                            device=state["_mean.weight"].device)
        state["_mean.weight"] = G @ state["_mean.weight"]
        state["_mean.bias"] = G @ state["_mean.bias"]
    return state


def latent_moments(Z):
    """Per-bin latent mean and covariance, e.g. of the source population's training latents."""
    Z = np.asarray(Z, dtype=np.float64)
    return {"mean": Z.mean(axis=0), "cov": np.cov(Z, rowvar=False)}


def fit_readin(X_train, device, seed, critic_arm, source_critic=None, cfg=BENCHMARK,
               epochs=None, moment_match=0.0, source_moments=None, init_encoder_state=None):
    """
    Train a new linear read-in for one population against a pretrained, random or fresh critic.

    Parameters
    ----------
    X_train : array (n_bins, n_units)
        Preprocessed training segment of the new population (neural data only).
    critic_arm : {"pretrained", "random", "fresh"}
        See the module docstring. "pretrained" and "random" need ``source_critic``.
    moment_match : float
        Optional weight on matching batch latent mean/covariance to ``source_moments``
        (neural-only; 0 disables it).
    init_encoder_state : dict, optional
        Read-in initialization (e.g. from ``transformed_encoder_state``) used in place of
        the random one; the same state should be passed to every arm being compared.

    Returns
    -------
    model : CPIC whose encoder is the new read-in and whose critic is the arm's critic
    history : per-epoch mean I_predictive / I_compress over full batches
    """
    if critic_arm not in CRITIC_ARMS:
        raise ValueError(f"critic_arm must be one of {CRITIC_ARMS}; got {critic_arm!r}")
    if critic_arm != "fresh" and source_critic is None:
        raise ValueError(f"critic_arm={critic_arm!r} needs source_critic")
    if moment_match > 0 and source_moments is None:
        raise ValueError("moment_match > 0 needs source_moments")
    epochs = cfg["epochs"] if epochs is None else epochs
    T, batch_size = cfg["T"], cfg["batch_size"]

    # Same seed -> identical read-in initialization in every arm.
    seed_everything(seed)
    model = make_cpic(X_train.shape[1], device, cfg)
    if init_encoder_state is not None:
        model.encoder.load_state_dict(init_encoder_state)
    if critic_arm == "pretrained":
        model.critic = freeze(copy.deepcopy(source_critic)).to(device)
    elif critic_arm == "random":
        model.critic = freeze(random_critic_like(source_critic, seed + 10_000)).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg["lr"])

    X = torch.as_tensor(np.asarray(X_train), dtype=torch.float32, device=device)
    n_windows = len(X) - 2 * T
    n_batches = n_windows // batch_size
    if n_batches == 0:
        raise ValueError(f"Need at least {batch_size} training windows; got {n_windows}")
    offsets = torch.arange(2 * T, device=device)
    generator = torch.Generator().manual_seed(seed)
    if moment_match > 0:
        target_mean = torch.as_tensor(source_moments["mean"], dtype=torch.float32, device=device)
        target_cov = torch.as_tensor(source_moments["cov"], dtype=torch.float32, device=device)

    history = []
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(n_windows, generator=generator)[:n_batches * batch_size]
        sums = torch.zeros(3, device=device)
        for idx in order.view(n_batches, batch_size):
            windows = X[idx.to(device)[:, None] + offsets]
            loss, i_compress, i_predictive = model(windows[:, :T], windows[:, T:])
            if moment_match > 0:
                z = model.encoder.get_mean(windows[:, :T]).reshape(-1, cfg["ydim"])
                zc = z - z.mean(dim=0)
                cov = zc.T @ zc / (len(z) - 1)
                loss = loss + moment_match * (((z.mean(dim=0) - target_mean) ** 2).sum()
                                              + ((cov - target_cov) ** 2).sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sums += torch.stack([loss.detach(), i_compress.detach().reshape(()),
                                 i_predictive.detach()])
        mean_loss, mean_compress, mean_predictive = (sums / n_batches).tolist()
        history.append({"epoch": epoch, "loss": mean_loss, "I_compress": mean_compress,
                        "I_predictive": mean_predictive})
    return model, history


def population_partitions(n_units, overlap, rng):
    """
    Split ``n_units`` simultaneously recorded units into two equal-size populations.

    ``overlap`` is the fraction of each population shared with the other: 0 gives
    disjoint halves, 1 gives two copies of the full population.
    """
    if not 0.0 <= overlap <= 1.0:
        raise ValueError(f"overlap must be in [0, 1]; got {overlap}")
    size = int(np.floor(n_units / (2.0 - overlap)))
    n_shared = int(round(overlap * size))
    order = rng.permutation(n_units)
    shared = order[:n_shared]
    only_1 = order[n_shared:size]
    only_2 = order[size:2 * size - n_shared]
    return np.sort(np.concatenate([shared, only_1])), np.sort(np.concatenate([shared, only_2]))
