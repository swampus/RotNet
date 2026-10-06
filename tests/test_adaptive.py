import pytest
import torch
from torch import nn

from rotnet.models import AdaptiveRotationNet
from rotnet.train import classification_loss


class CountStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, x):
        self.batches.append(len(x))
        return x + 1


class ScriptedAdaptive(AdaptiveRotationNet):
    def project(self, x):
        return x


def scripted_model():
    model = ScriptedAdaptive(hidden_dim=8)
    model.core.stages = nn.ModuleList([CountStage() for _ in range(3)])
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.weight[0, 0] = 2
    return model.eval()


def test_early_exit_skips_later_stage_computation():
    model = scripted_model()
    output = model.infer(torch.full((4, 8), 2.0), threshold=0.9)
    assert output.exit_stage.tolist() == [1] * 4
    assert output.prediction.tolist() == [0] * 4
    assert (output.confidence >= 0.9).all()
    assert [stage.batches for stage in model.core.stages] == [[4], [], []]


def test_never_confident_reaches_eighth_stage():
    model = AdaptiveRotationNet().eval()
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
    output = model.infer(torch.randn(3, 1, 28, 28), threshold=0.99)
    assert output.exit_stage.tolist() == [8, 8, 8]
    torch.testing.assert_close(output.confidence, torch.full((3,), 0.1))


def test_mixed_exits_keep_sample_order_and_return_chosen_logits():
    model = scripted_model()
    images = torch.zeros(2, 8)
    images[:, 0] = torch.tensor([2.0, -1.0])
    output = model.infer(images, threshold=0.9)
    assert output.exit_stage.tolist() == [1, 3]
    assert [stage.batches for stage in model.core.stages] == [[2], [1], [1]]
    expected = torch.zeros(2, 10)
    expected[:, 0] = torch.tensor([6.0, 4.0])
    torch.testing.assert_close(output.logits, expected)
    torch.testing.assert_close(output.confidence, expected.softmax(dim=-1).amax(dim=-1))


def test_hard_inference_matches_all_stage_reference():
    torch.manual_seed(5)
    model = AdaptiveRotationNet(hidden_dim=8).eval()
    images = torch.randn(6, 1, 28, 28)
    with torch.no_grad():
        all_logits = model.forward_all(images)
    for threshold in (0.0, 0.15, 0.90, 1.0):
        confidence = all_logits.softmax(dim=-1).amax(dim=-1)
        eligible = confidence >= threshold
        eligible[:, -1] = True
        chosen = eligible.int().argmax(dim=-1)
        expected = all_logits[torch.arange(6), chosen]
        output = model.infer(images, threshold)
        assert torch.equal(output.exit_stage, chosen + 1)
        torch.testing.assert_close(output.logits, expected)


def test_training_loss_is_equal_mean_and_gradients_reach_all_stages():
    torch.manual_seed(4)
    model = AdaptiveRotationNet(hidden_dim=8)
    targets = torch.tensor([1, 2, 3, 4])
    logits = model.forward_all(torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, 3, 10)
    expected = torch.stack([nn.functional.cross_entropy(logits[:, i], targets) for i in range(3)]).mean()
    loss = classification_loss(logits, targets)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for stage in model.core.stages:
        assert stage.rotation.theta.grad is not None
        assert stage.rotation.theta.grad.abs().sum() > 0


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        AdaptiveRotationNet(threshold=threshold)


def test_inference_requires_eval_mode():
    with pytest.raises(RuntimeError, match="eval"):
        AdaptiveRotationNet().infer(torch.zeros(1, 784))
