import pytest
import torch
from torch import nn

from rotnet.metrics import parameter_counts
from rotnet.models import make_model
from rotnet.phase2.models import VARIANTS, arithmetic, controlled_model
from rotnet.train import seed_everything


@pytest.mark.parametrize("name", ["dense", "low-rank", "rotation", "adaptive-rotation"])
def test_phase2_default_is_unchanged_phase1_model(name):
    seed_everything(42)
    original = make_model(name)
    seed_everything(42)
    replica = controlled_model(name)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, replica.state_dict()[key], atol=0, rtol=0)
    x = torch.randn(4, 1, 28, 28)
    torch.testing.assert_close(original(x), replica(x), atol=0, rtol=0)


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_stage_prefix_initialization_and_parameter_count(count):
    seed_everything(44)
    full = controlled_model("rotation")
    seed_everything(44)
    prefix = controlled_model("rotation", stages=count)
    assert len(prefix.core.stages) == count
    assert prefix.core.strides == [1 << i for i in range(count)]
    assert parameter_counts(prefix)["core_trainable_parameters"] == 640 * count
    assert arithmetic(prefix)["core_multiply_add_ops_per_sample"] == 1280 * count
    for a, b in zip(full.core.stages, prefix.core.stages):
        torch.testing.assert_close(a.rotation.theta, b.rotation.theta, atol=0, rtol=0)


@pytest.mark.parametrize("variant", ["identity", "random-fixed"])
def test_frozen_angles_remain_fixed_while_affine_parameters_train(variant):
    model = controlled_model("rotation", variant=variant)
    before = [stage.rotation.theta.clone() for stage in model.core.stages]
    loss = nn.functional.cross_entropy(model(torch.randn(3, 1, 28, 28)), torch.tensor([1, 2, 3]))
    loss.backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    optimizer.step()
    for stage, theta in zip(model.core.stages, before):
        assert not stage.rotation.theta.requires_grad
        assert stage.rotation.theta.grad is None
        assert stage.scale.grad is not None and stage.bias.grad is not None
        torch.testing.assert_close(stage.rotation.theta, theta, atol=0, rtol=0)
        if variant == "identity":
            assert torch.equal(theta, torch.zeros_like(theta))
    counts = parameter_counts(model)
    assert counts["core_trainable_parameters"] == 4096
    assert counts["core_total_parameters"] == 5120


def test_identity_and_no_mixing_depth_control_have_identical_function():
    seed_everything(42)
    identity = controlled_model("rotation", variant="identity")
    seed_everything(42)
    coordinate = controlled_model("rotation", variant="coordinate-only")
    x = torch.randn(4, 1, 28, 28)
    torch.testing.assert_close(identity(x), coordinate(x), atol=0, rtol=0)
    assert arithmetic(coordinate)["core_multiply_add_ops_per_sample"] == 4096
    assert parameter_counts(coordinate)["core_total_parameters"] == 4096


@pytest.mark.parametrize("variant", VARIANTS)
def test_component_control_forward_and_no_dense_core(variant):
    model = controlled_model("rotation", variant=variant)
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)
    assert not any(tuple(p.shape) == (256, 256) for p in model.core.parameters())
    gelus = sum(isinstance(m, nn.GELU) for m in model.core.modules())
    assert gelus == (1 if variant == "final-gelu" else 8)
    if variant == "no-affine":
        assert parameter_counts(model)["core_trainable_parameters"] == 1024
        assert arithmetic(model)["core_multiply_add_ops_per_sample"] == 6144


@pytest.mark.parametrize("name,total", [("dense",68362), ("low-rank",19210), ("rotation",7690), ("adaptive-rotation",7690)])
def test_direct16_removes_projection_and_initial_activation(name,total):
    model = controlled_model(name, representation="16x16")
    assert isinstance(model.projection, nn.Identity)
    assert isinstance(model.activation, nn.Identity)
    assert parameter_counts(model)["total_parameters"] == total
    x = torch.randn(2, 1, 16, 16)
    torch.testing.assert_close(model.project(x), x.flatten(1), atol=0, rtol=0)
    assert model(x).shape == (2, 10)
    assert arithmetic(model)["projection_multiply_add_ops_per_sample"] == 0


def test_repeated_heads_and_full_depth_arithmetic():
    model = controlled_model("adaptive-rotation", representation="16x16")
    assert arithmetic(model,2)["whole_model_multiply_add_ops_per_sample"] == 12800
    assert arithmetic(model,8,final_head_only=True)["whole_model_multiply_add_ops_per_sample"] == 15360


def test_projection_only_control_has_no_core():
    model = controlled_model("projection-only")
    assert parameter_counts(model)["core_trainable_parameters"] == 0
    assert arithmetic(model)["core_multiply_add_ops_per_sample"] == 0
    assert model(torch.zeros(1, 1, 28, 28)).shape == (1, 10)
