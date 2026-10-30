"""
What can a frozen CPIC critic align?

A trained critic is a scoring function on (past, future) latent windows, not an
identifiable dynamics model. Before using it to place a new population's latents into
a source coordinate system, these tools measure:

    critic_bound_under_transform   frozen-critic InfoNCE after a controlled latent map
                                   z -> z G^T + b (same map at every time bin)
    invariance_profile             that bound over families of maps (I1): any map the
                                   critic scores like the identity cannot be resolved
    objective_sensitivity          gradient energy of the InfoNCE bound per latent
                                   direction: directions the critic ignores are
                                   unconstrained by critic-anchored alignment
    recover_linear_map             self-recovery (I2): starting from a random map, can
                                   maximizing frozen-critic InfoNCE find the identity?

Everything here is neural-only.
"""

import numpy as np
import torch

from .evaluation import _window_batches, encode
from .mi import infonce_lower_bound


def _latent_windows(Z, idx, T, offsets):
    windows = Z[idx[:, None] + offsets]
    return windows[:, :T].reshape(len(idx), -1), windows[:, T:].reshape(len(idx), -1)


def critic_bound_under_transform(model, X, G=None, b=None, critic=None, batch_size=512,
                                 seed=0, max_batches=None, Z=None):
    """
    Frozen-critic InfoNCE lower bound on mean latents of ``X`` after z -> z G^T + b.

    ``critic`` defaults to ``model.critic``; pass another model's critic to score this
    model's latents under it. ``Z`` may be passed to skip re-encoding.
    """
    critic = model.critic if critic is None else critic
    T, device = model.T, model.device
    Z = torch.as_tensor(encode(model, X) if Z is None else Z, dtype=torch.float32, device=device)
    if G is not None:
        Z = Z @ torch.as_tensor(np.asarray(G), dtype=torch.float32, device=device).T
    if b is not None:
        Z = Z + torch.as_tensor(np.asarray(b), dtype=torch.float32, device=device)
    batches = _window_batches(len(Z) - 2 * T, batch_size, seed, max_batches)
    offsets = torch.arange(2 * T, device=device)
    bounds = []
    was_training = critic.training
    critic.eval()
    try:
        with torch.no_grad():
            for idx in batches:
                z_past, z_future = _latent_windows(Z, torch.as_tensor(idx, device=device), T, offsets)
                bounds.append(float(infonce_lower_bound(critic(z_past, z_future))))
    finally:
        critic.train(was_training)
    return float(np.mean(bounds))


def haar_orthogonal(d, rng):
    Q, R = np.linalg.qr(rng.normal(size=(d, d)))
    return Q * np.sign(np.diag(R))


def plane_rotation(U, theta):
    """Rotation by ``theta`` inside the plane spanned by the orthonormal columns of U (d x 2)."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return np.eye(U.shape[0]) - U @ U.T + U @ R @ U.T


def oscillatory_planes(Z_train, n_planes):
    """
    Orthonormal bases of the oscillatory planes of a least-squares linear model
    z_{t+1} = A z_t fit on (centered) training latents, slowest-decaying first.
    """
    Z_train = np.asarray(Z_train, dtype=np.float64)  # float32 latents give only ~1e-7-orthogonal planes
    Zc = Z_train - Z_train.mean(axis=0)
    A = np.linalg.lstsq(Zc[:-1], Zc[1:], rcond=None)[0].T
    eigvals, eigvecs = np.linalg.eig(A)
    planes, seen = [], set()
    for i in np.argsort(-np.abs(eigvals)):
        if abs(eigvals[i].imag) < 1e-8:
            continue
        key = (round(abs(eigvals[i]), 10), round(abs(eigvals[i].imag), 10))
        if key in seen:  # skip the conjugate partner
            continue
        seen.add(key)
        U, _ = np.linalg.qr(np.stack([eigvecs[:, i].real, eigvecs[:, i].imag], axis=1))
        planes.append({"U": U, "eigenvalue": complex(eigvals[i])})
        if len(planes) == n_planes:
            break
    return planes


def transformation_suite(Z_train, rng, n_draws=3, angles_deg=(5, 10, 20, 45, 90)):
    """Controlled latent maps (G, b) with labels; G acts identically on every time bin."""
    Z_train = np.asarray(Z_train, dtype=np.float64)
    d = Z_train.shape[1]
    std = Z_train.std(axis=0)
    out = [{"family": "identity", "param": 0.0, "draw": 0, "G": np.eye(d), "b": None}]

    def add(family, param, draw, G, b=None):
        out.append({"family": family, "param": float(param), "draw": draw, "G": G, "b": b})

    for draw in range(n_draws):
        signs = rng.choice([-1.0, 1.0], size=d)
        signs[rng.integers(d)] = -1.0  # at least one flip
        add("sign_flip", 0, draw, np.diag(signs))
        add("permutation", 0, draw, np.eye(d)[rng.permutation(d)])
        add("orthogonal_haar", 0, draw, haar_orthogonal(d, rng))
        add("random_linear", 0, draw, rng.normal(size=(d, d)) / np.sqrt(d))
        for sigma in (0.1, 0.3):
            add("scale_anisotropic", sigma, draw, np.diag(np.exp(rng.normal(0, sigma, size=d))))
        for c in (0.5, 1.0, 2.0):
            u = rng.normal(size=d)
            add("translation", c, draw, np.eye(d), c * std * u / np.linalg.norm(u))
        for angle in angles_deg:
            theta = np.deg2rad(angle)
            basis = haar_orthogonal(d, rng)
            G = np.eye(d)
            for k in range(d // 2):  # rotate every plane of a random orthonormal basis
                G = plane_rotation(basis[:, 2 * k:2 * k + 2], theta) @ G
            add("rotation_global", angle, draw, G)
    for scale in (0.5, 0.8, 1.25, 2.0):
        add("scale_isotropic", scale, 0, scale * np.eye(d))
    for n_planes in (1, 3):
        planes = oscillatory_planes(Z_train, n_planes)
        if len(planes) < n_planes:
            continue
        for angle in angles_deg:
            G = np.eye(d)
            for plane in planes:
                G = plane_rotation(plane["U"], np.deg2rad(angle)) @ G
            add(f"rotation_dynamics_top{n_planes}", angle, 0, G)
    return out


def invariance_profile(model, X_eval, Z_train, critic=None, seed=0, batch_size=512,
                       max_batches=None, n_draws=3):
    """
    Frozen-critic bound for each map in ``transformation_suite`` (I1).

    Returns rows with ``bound`` and ``delta`` = bound(identity) - bound(map); delta near 0
    means the critic cannot distinguish that map from the identity.
    """
    rng = np.random.default_rng(seed)
    Z_eval = encode(model, X_eval)
    rows = []
    for t in transformation_suite(Z_train, rng, n_draws=n_draws):
        bound = critic_bound_under_transform(model, None, t["G"], t["b"], critic=critic,
                                             batch_size=batch_size, seed=seed,
                                             max_batches=max_batches, Z=Z_eval)
        rows.append({"family": t["family"], "param": t["param"], "draw": t["draw"],
                     "bound": bound})
    identity = rows[0]["bound"]
    for row in rows:
        row["delta"] = identity - row["bound"]
    return rows


def objective_sensitivity(model, X, Z_train, critic=None, batch_size=512, seed=0, max_batches=4):
    """
    Mean squared gradient of the frozen-critic InfoNCE bound w.r.t. the latents, summed over
    time bins and past/future, per latent coordinate and per principal axis of ``Z_train``.
    """
    critic = model.critic if critic is None else critic
    T, d, device = model.T, model.ydim, model.device
    Z = torch.as_tensor(encode(model, X), dtype=torch.float32, device=device)
    _, sv, Vt = np.linalg.svd(Z_train - Z_train.mean(axis=0), full_matrices=False)
    V = torch.as_tensor(Vt.T, dtype=torch.float32, device=device)
    offsets = torch.arange(2 * T, device=device)
    coord, principal = torch.zeros(d, device=device), torch.zeros(d, device=device)
    batches = _window_batches(len(Z) - 2 * T, batch_size, seed, max_batches)
    for idx in batches:
        z_past, z_future = _latent_windows(Z, torch.as_tensor(idx, device=device), T, offsets)
        z_past.requires_grad_(True)
        z_future.requires_grad_(True)
        bound = infonce_lower_bound(critic(z_past, z_future))
        g_past, g_future = torch.autograd.grad(bound, (z_past, z_future))
        grads = torch.cat([g_past.reshape(-1, T, d), g_future.reshape(-1, T, d)], dim=1)
        coord += (grads ** 2).sum(dim=(0, 1))
        principal += ((grads @ V) ** 2).sum(dim=(0, 1))
    coord, principal = coord / len(batches), principal / len(batches)
    return {"energy_by_coordinate": coord.cpu().numpy(),
            "energy_by_principal_axis": principal.cpu().numpy(),
            "principal_axis_variance": sv ** 2 / (len(Z_train) - 1)}


def recover_linear_map(model, Z_train, Z_eval, critic=None, parametrization="orthogonal",
                       seed=0, steps=500, lr=1e-2, batch_size=512):
    """
    Self-recovery (I2): learn z -> z G^T + b on the model's own latents by maximizing the
    frozen-critic bound, starting from a Haar-random rotation. The correct answer is G = I.

    parametrization="orthogonal" keeps G a rotation (G = expm(S - S^T) G0), which asks only
    whether the critic pins down orientation; "linear" lets G (and b) be any matrix.
    """
    critic = model.critic if critic is None else critic
    T, d, device = model.T, model.ydim, model.device
    rng = np.random.default_rng(seed)
    G0 = torch.as_tensor(haar_orthogonal(d, rng), dtype=torch.float32, device=device)
    Z = torch.as_tensor(Z_train, dtype=torch.float32, device=device)
    S = torch.zeros(d, d, device=device, requires_grad=True)
    W = G0.clone().requires_grad_(True)
    b = torch.zeros(d, device=device, requires_grad=True)
    params = [S] if parametrization == "orthogonal" else [W, b]
    if parametrization not in ("orthogonal", "linear"):
        raise ValueError(f"Unknown parametrization {parametrization!r}")
    optimizer = torch.optim.Adam(params, lr=lr)
    critic_params = list(critic.parameters())
    requires = [p.requires_grad for p in critic_params]
    for p in critic_params:
        p.requires_grad_(False)

    def current_map():
        if parametrization == "orthogonal":
            return torch.matrix_exp(S - S.T) @ G0, torch.zeros(d, device=device)
        return W, b

    offsets = torch.arange(2 * T, device=device)
    generator = torch.Generator().manual_seed(seed)
    n_windows = len(Z) - 2 * T
    history = []
    try:
        for step in range(steps):
            idx = torch.randperm(n_windows, generator=generator)[:batch_size].to(device)
            G, bias = current_map()
            z_past, z_future = _latent_windows(Z @ G.T + bias, idx, T, offsets)
            bound = infonce_lower_bound(critic(z_past, z_future))
            optimizer.zero_grad()
            (-bound).backward()
            optimizer.step()
            if step % 50 == 0 or step == steps - 1:
                history.append({"step": step, "bound": float(bound.detach())})
    finally:
        for p, r in zip(critic_params, requires):
            p.requires_grad_(r)

    with torch.no_grad():
        G, bias = current_map()
        G, bias = G.cpu().numpy(), bias.cpu().numpy()
    Z_eval = np.asarray(Z_eval, dtype=np.float64)
    Z_mapped = Z_eval @ G.T + bias
    residual = ((Z_mapped - Z_eval) ** 2).sum() / ((Z_eval - Z_eval.mean(axis=0)) ** 2).sum()
    singular = np.linalg.svd(G, compute_uv=False)
    return {"G": G, "b": bias, "parametrization": parametrization, "seed": int(seed),
            "distance_from_identity": float(np.linalg.norm(G - np.eye(d)) / np.sqrt(d)),
            "initial_distance_from_identity": float(
                np.linalg.norm(G0.cpu().numpy() - np.eye(d)) / np.sqrt(d)),
            "direct_r2_vs_identity": float(1.0 - residual),
            "singular_values_min_max": [float(singular.min()), float(singular.max())],
            "history": history}
