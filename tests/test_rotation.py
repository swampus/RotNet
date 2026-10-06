import pytest
import torch

from rotnet.models.rotation import PairwiseRotation, RotationNet, butterfly_strides, pairing_indices


@pytest.mark.parametrize("stride,left,right", [
    (1, [0, 2, 4, 6], [1, 3, 5, 7]),
    (2, [0, 1, 4, 5], [2, 3, 6, 7]),
    (4, [0, 1, 2, 3], [4, 5, 6, 7]),
])
def test_pairing_n8(stride, left, right):
    actual_left, actual_right = pairing_indices(8, stride)
    assert actual_left.tolist() == left
    assert actual_right.tolist() == right


@pytest.mark.parametrize("n", [2, 8, 256])
def test_each_coordinate_occurs_once_per_stage(n):
    for stride in butterfly_strides(n):
        left, right = pairing_indices(n, stride)
        assert torch.cat((left, right)).sort().values.tolist() == list(range(n))
        assert torch.equal(left ^ stride, right)


@pytest.mark.parametrize("stride", [1, 2, 4, 8, 16, 32, 64, 128])
def test_pure_rotation_preserves_l2_norm(stride):
    torch.manual_seed(7)
    rotation = PairwiseRotation(256, stride).double()
    x = torch.randn(5, 256, dtype=torch.float64)
    torch.testing.assert_close(rotation(x).norm(dim=-1), x.norm(dim=-1), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_theta_order_matches_pairing_and_gradients_reach_theta(stride):
    rotation = PairwiseRotation(8, stride).double()
    with torch.no_grad():
        rotation.theta.copy_(torch.tensor([0.1, -0.3, 0.7, 1.2], dtype=torch.float64))
    x = torch.arange(1, 9, dtype=torch.float64).reshape(1, 8).requires_grad_()
    left, right = pairing_indices(8, stride)
    expected = torch.empty_like(x)
    for i, (a, b) in enumerate(zip(left.tolist(), right.tolist())):
        cos, sin = rotation.theta[i].cos(), rotation.theta[i].sin()
        expected[:, a] = x[:, a] * cos - x[:, b] * sin
        expected[:, b] = x[:, a] * sin + x[:, b] * cos
    actual = rotation(x)
    torch.testing.assert_close(actual, expected)
    (actual * torch.arange(8, dtype=torch.float64)).sum().backward()
    assert rotation.theta.grad is not None
    assert torch.isfinite(rotation.theta.grad).all()
    assert (rotation.theta.grad.abs() > 1e-8).all()
    assert x.grad is not None


@pytest.mark.parametrize("shape", [(8,), (1, 8), (5, 8), (2, 3, 8)])
def test_batches_equal_individual_rotations(shape):
    torch.manual_seed(11)
    rotation = PairwiseRotation(8, 2)
    x = torch.randn(*shape)
    actual = rotation(x)
    expected = torch.stack([rotation(row) for row in x.reshape(-1, 8)]).reshape_as(x)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)


def test_butterfly_connects_all_inputs_structurally():
    # Receptive-field connectivity of the pair graph, independent of trained angles.
    reach = [{i} for i in range(256)]
    for stride in butterfly_strides(256):
        left, right = pairing_indices(256, stride)
        for a, b in zip(left.tolist(), right.tolist()):
            merged = reach[a] | reach[b]
            reach[a], reach[b] = merged, merged
    assert all(len(r) == 256 for r in reach)


def test_no_dense_hidden_matrix_or_linear_in_rotation_core():
    model = RotationNet()
    assert len(model.core.stages) == 8
    assert not any(tuple(p.shape) == (256, 256) for p in model.parameters())
    assert all(p.ndim == 1 for p in model.core.parameters())
    assert not any(isinstance(module, torch.nn.Linear) for module in model.core.modules())
    assert model(torch.randn(3, 1, 28, 28)).shape == (3, 10)


@pytest.mark.parametrize("n,stride", [(0, 1), (7, 1), (8, 3), (8, 8)])
def test_invalid_pairing_rejected(n, stride):
    with pytest.raises(ValueError):
        pairing_indices(n, stride)
