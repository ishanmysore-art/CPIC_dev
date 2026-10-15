"""Read-only measurements for the neural/mask input ablation."""

import numpy as np
import os
import torch


AVAILABILITY_FEATURES = ("actual", "zeros", "ones")


def result_filename(saved_root, kernel, availability_mask, availability_features,
                    beta, ydim, seed, neuron_dropout_p, diagnose_mlp=False,
                    diagnose_ablation=False, fixed_epochs=False):
    """A distinct, non-overwriting result path for every input representation."""
    if availability_features not in AVAILABILITY_FEATURES:
        raise ValueError(f"Unknown availability-features mode: {availability_features}")
    if not availability_mask and availability_features != "actual":
        raise ValueError("zeros/ones availability features require --availability-mask")
    feature_tag = f"_availfeat{availability_features}" if availability_mask else ""
    run_tag = f"{kernel.lower()}_avail{int(availability_mask)}{feature_tag}_beta{beta:g}"
    return os.path.join(
        saved_root,
        f"result_{run_tag}_ydim{ydim}_seed{seed}"
        f"_traindrop{int(round(neuron_dropout_p * 100)):03d}"
        f"{'_fixed' if fixed_epochs else ''}"
        f"{'_diagmlp' if diagnose_mlp else ''}"
        f"{'_diagablation' if diagnose_ablation else ''}.pkl",
    )


def append_features_torch(masked_activity, neuron_mask, mode):
    """Append features without changing the mask used on neural activity."""
    if mode not in AVAILABILITY_FEATURES:
        raise ValueError(f"Unknown availability-features mode: {mode}")
    features = neuron_mask if mode == "actual" else torch.full_like(
        neuron_mask, 0.0 if mode == "zeros" else 1.0)
    return torch.cat((masked_activity, features.expand_as(masked_activity)), dim=-1)


def build_encoder_input(activity, neuron_mask, availability_mask=False,
                        availability_features="actual"):
    """Evaluation counterpart of the training input construction."""
    masked = activity * neuron_mask
    if not availability_mask:
        return masked
    if availability_features not in AVAILABILITY_FEATURES:
        raise ValueError(f"Unknown availability-features mode: {availability_features}")
    features = neuron_mask if availability_features == "actual" else np.full_like(
        neuron_mask, 0.0 if availability_features == "zeros" else 1.0)
    return np.concatenate((masked, np.broadcast_to(features, activity.shape).astype(
        activity.dtype, copy=False)), axis=-1)


def first_layer_stats(inputs, linear, neural_xdim):
    """Decompose a mean-network first-layer preactivation; no forward pass or RNG."""
    with torch.no_grad():
        flat = inputs.detach().reshape(-1, inputs.shape[-1])
        weight = linear.weight.detach()
        neural = torch.nn.functional.linear(flat[:, :neural_xdim],
                                             weight[:, :neural_xdim])
        extra = torch.nn.functional.linear(flat[:, neural_xdim:],
                                            weight[:, neural_xdim:])
        base = neural + linear.bias.detach()
        full = base + extra
        rms = lambda x: float(x.square().mean().sqrt().item())
        return {
            "neural_contribution_rms": rms(neural),
            "appended_contribution_rms": rms(extra),
            "preactivation_rms": rms(full),
            "relu_active_fraction": float((full > 0).float().mean().item()),
            "relu_active_without_appended_fraction": float((base > 0).float().mean().item()),
            "relu_gate_change_from_appended_fraction": float(((full > 0) != (base > 0)).float().mean().item()),
        }


def paired_first_layer_stats(clean_inputs, masked_inputs, linear, neural_xdim):
    """Fraction of ReLU gates changed between aligned clean and masked inputs."""
    with torch.no_grad():
        clean = torch.nn.functional.linear(clean_inputs.detach(), linear.weight.detach(), linear.bias.detach())
        masked = torch.nn.functional.linear(masked_inputs.detach(), linear.weight.detach(), linear.bias.detach())
        result = first_layer_stats(masked_inputs, linear, neural_xdim)
        result["relu_gate_change_clean_to_masked_fraction"] = float(
            ((clean > 0) != (masked > 0)).float().mean().item())
        result["preactivation_displacement_rms"] = float(
            (masked - clean).square().mean().sqrt().item())
        return result


def latent_pair_stats(clean, masked):
    """Per-dimension statistics of aligned deterministic latent means."""
    clean, masked = np.asarray(clean), np.asarray(masked)
    if clean.shape != masked.shape:
        raise ValueError("Clean and masked latents must be aligned")
    delta = masked - clean
    return {
        "clean_mean": clean.mean(axis=0), "masked_mean": masked.mean(axis=0),
        "clean_variance": clean.var(axis=0), "masked_variance": masked.var(axis=0),
        "clean_min": clean.min(axis=0), "clean_max": clean.max(axis=0),
        "masked_min": masked.min(axis=0), "masked_max": masked.max(axis=0),
        "paired_displacement_mean": delta.mean(axis=0),
        "paired_displacement_rms": np.sqrt(np.mean(delta ** 2, axis=0)),
        "paired_displacement_norm_mean": float(np.linalg.norm(delta, axis=1).mean()),
    }


def clean_decoder_stats(model, clean_design):
    """Inspect an already-fitted decoder and its clean centered design matrix."""
    design = np.asarray(clean_design, dtype=np.float64)
    singular = np.linalg.svd(design - design.mean(axis=0), compute_uv=False,
                             full_matrices=False)
    coefficients = np.atleast_2d(model.coef_)
    return {
        "coefficient_norm_per_output": np.linalg.norm(coefficients, axis=1),
        "coefficient_spectral_norm": float(np.linalg.norm(coefficients, ord=2)),
        "singular_values": singular,
        "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf"),
    }


def prediction_pair_stats(clean_prediction, masked_prediction, targets):
    """Error decomposition for the same fitted decoder on aligned test targets."""
    as_outputs = lambda value: np.asarray(value).reshape(-1, 1) if np.asarray(value).ndim == 1 else np.asarray(value)
    clean = as_outputs(clean_prediction)
    masked = as_outputs(masked_prediction)
    target = as_outputs(targets)
    if clean.shape != masked.shape or clean.shape != target.shape:
        raise ValueError("Predictions and targets must be aligned")
    delta = masked - clean
    sst = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    clean_sse = ((target - clean) ** 2).sum(axis=0)
    masked_sse = ((target - masked) ** 2).sum(axis=0)
    r2 = lambda sse: np.where(sst > 0, 1 - sse / np.where(sst > 0, sst, 1),
                              (sse == 0).astype(float))
    return {
        "clean_prediction_min": clean.min(axis=0), "clean_prediction_max": clean.max(axis=0),
        "masked_prediction_min": masked.min(axis=0), "masked_prediction_max": masked.max(axis=0),
        "prediction_displacement_mean": delta.mean(axis=0),
        "prediction_displacement_rms": np.sqrt(np.mean(delta ** 2, axis=0)),
        "clean_sse": clean_sse, "masked_sse": masked_sse, "sst": sst,
        "clean_r2_per_output": r2(clean_sse), "masked_r2_per_output": r2(masked_sse),
    }
