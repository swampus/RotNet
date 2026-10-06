import pytest
import torch

from rotnet.data import make_loader
from rotnet.models import AdaptiveRotationNet, RotationNet
from rotnet.phase2.runtime import CachedRotation, optimize_inference, verify_equivalence


@pytest.mark.parametrize("adaptive",[False,True])
def test_optimization_matches_outputs_and_leaves_original_unchanged(adaptive):
    torch.manual_seed(7)
    model = (AdaptiveRotationNet if adaptive else RotationNet)(hidden_dim=8).eval()
    before = {key:value.clone() for key,value in model.state_dict().items()}
    optimized = optimize_inference(model)
    dataset = torch.utils.data.TensorDataset(torch.randn(9,1,28,28),torch.arange(9)%10,torch.arange(9))
    loader = make_loader(dataset,4,42,False)
    for threshold in (0,0.15,0.9,1):
        result = verify_equivalence(model,optimized,loader,torch.device("cpu"),threshold)
        assert result["samples_checked"] == 9
        assert result["max_absolute_logit_error"] == 0
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    assert all(p.requires_grad for p in model.parameters())
    assert not any(p.requires_grad for p in optimized.parameters())
    assert all(isinstance(stage.rotation,CachedRotation) for stage in optimized.core.stages)


def test_cache_rejects_gradient_and_training_paths():
    optimized = optimize_inference(RotationNet(hidden_dim=8))
    images = torch.randn(2,1,28,28)
    with pytest.raises(RuntimeError,match="inference-only"):
        optimized(images)
    optimized.train()
    with torch.no_grad(),pytest.raises(RuntimeError,match="inference-only"):
        optimized(images)


def test_direct16_cached_adaptive_supports_mixed_inputs():
    from rotnet.phase2.models import controlled_model
    original = controlled_model("adaptive-rotation",representation="16x16").eval()
    optimized = optimize_inference(original)
    x = torch.randn(5,1,16,16)
    with torch.inference_mode():
        for threshold in (0,0.2,0.9,1):
            a,b = original.infer(x,threshold),optimized.infer(x,threshold)
            assert torch.equal(a.exit_stage,b.exit_stage)
            assert torch.equal(a.prediction,b.prediction)
            torch.testing.assert_close(a.confidence,b.confidence,rtol=0,atol=0)
            torch.testing.assert_close(a.logits,b.logits,rtol=0,atol=0)
