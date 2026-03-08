import torch
from cpic.models import StructuredEncoder


# Run these tests from the repo root with pytest -q
def make_input(batch_size: int = 2, T: int = 5, D: int = 8) -> torch.Tensor:
    """Create a dummy input of shape (batch, T, D)."""
    return torch.randn(batch_size, T, D)


def build_encoder(
    encoder_type: str,
    input_dim: int = 8,
    hidden_dim: int = 16,
    output_dim: int = 4,
    T: int = 5,
    deterministic: bool = False,
    **encoder_kwargs,
) -> StructuredEncoder:
    """Helper to construct a StructuredEncoder for tests."""
    return StructuredEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        T=T,
        deterministic=deterministic,
        encoder_type=encoder_type,
        **encoder_kwargs,
    )


def _assert_mean_logvar_shapes(encoder_type: str, **encoder_kwargs) -> None:
    batch_size = 2
    T = 5
    D = 8
    M = 4
    x = make_input(batch_size=batch_size, T=T, D=D)
    encoder = build_encoder(
        encoder_type=encoder_type,
        input_dim=D,
        hidden_dim=16,
        output_dim=M,
        T=T,
        deterministic=False,
        **encoder_kwargs,
    )

    mean = encoder.get_mean(x)
    logvars = encoder.get_logvars(x)

    assert mean.shape == (batch_size, T, M)
    assert logvars.shape == (batch_size, T, M)


def test_linear_encoder_shapes():
    _assert_mean_logvar_shapes("linear")


def test_mlp_encoder_shapes():
    _assert_mean_logvar_shapes("mlp", n_layers=1, activation="relu")


def test_mlp2_encoder_shapes():
    _assert_mean_logvar_shapes("mlp2", activation="relu")


def test_conv_spatial_encoder_shapes():
    _assert_mean_logvar_shapes(
        "conv_spatial",
        n_layers=0,
        conv_kernel_size=3,
        conv_stride=1,
        conv_padding=1,
    )


def test_conv_spatiotemporal_encoder_shapes():
    _assert_mean_logvar_shapes(
        "conv_spatiotemporal",
        n_layers=0,
        conv_kernel_size=3,
        conv_stride=1,
        conv_padding=1,
    )


def test_conv_temporal_encoder_shapes():
    _assert_mean_logvar_shapes(
        "conv_temporal",
        n_layers=1,
        kernel_size_1d=3,
        activation="relu",
    )


def test_deterministic_encoder_same_mean():
    """Deterministic encoder should give identical means for the same input."""
    batch_size, T, D, M = 2, 5, 8, 4
    x = make_input(batch_size=batch_size, T=T, D=D)
    encoder = build_encoder(
        encoder_type="mlp",
        input_dim=D,
        hidden_dim=16,
        output_dim=M,
        T=T,
        deterministic=True,
        n_layers=1,
        activation="relu",
    )

    mean1 = encoder.get_mean(x)
    mean2 = encoder.get_mean(x)
    assert torch.allclose(mean1, mean2)


def test_encoder_registry_forward_passes():
    """All registered encoder types used in CPIC should build and run a forward pass."""
    encoder_types = [
        "linear",
        "mlp",
        "mlp2",
        "conv_spatial",
        "conv_spatiotemporal",
        "conv_temporal",
    ]

    batch_size, T, D, M = 2, 5, 8, 4
    x = make_input(batch_size=batch_size, T=T, D=D)

    for enc_type in encoder_types:
        encoder = build_encoder(
            encoder_type=enc_type,
            input_dim=D,
            hidden_dim=16,
            output_dim=M,
            T=T,
            deterministic=False,
        )
        mean, vars_ = encoder(x)
        assert mean.shape == (batch_size, T, M)
        assert vars_.shape == (batch_size, T, M)
