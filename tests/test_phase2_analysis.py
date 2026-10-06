import inspect
import math

import pytest
import torch
from torch.utils.data import TensorDataset

from rotnet.phase2.analysis import exits_from_logits, select_policy, stage_transitions
from rotnet.phase2.data import resize_direct, split_record, validation_split
from rotnet.phase2.report import summary
from rotnet.phase2.runner import FAMILIES, case_id, plan


def policy_fixture():
    logits = torch.tensor([[[4.,0.],[4.,0.],[4.,0.]],
                           [[0.,0.2],[4.,0.],[4.,0.]],
                           [[0.,4.],[0.,4.],[0.,4.]],
                           [[0.,4.],[0.,4.],[0.,4.]]])
    targets = torch.tensor([0,0,1,1])
    return logits, targets


def test_validation_threshold_respects_tolerance_and_minimizes_stages():
    logits, targets = policy_fixture()
    selection = select_policy(logits,targets,[0.5,0.9,1.0],tolerance_pp=0.1)
    assert selection["selected"]["threshold"] == 0.9
    assert selection["selected"]["validation_accuracy"] == 1
    assert selection["selected"]["validation_mean_stages"] == 1.25
    assert selection["selection_split"] == "validation"
    assert all("test" not in name for name in inspect.signature(select_policy).parameters)


def test_full_depth_is_guaranteed_fallback_if_all_thresholds_fail():
    logits, targets = policy_fixture()
    selection = select_policy(logits,targets,[0.5])
    assert selection["selected"]["policy"] == "full-depth"
    assert selection["selected"]["threshold"] is None
    assert selection["selected"]["validation_accuracy"] == 1


def test_tolerance_is_percentage_points_and_not_fraction():
    logits, targets = policy_fixture()
    assert select_policy(logits,targets,[0.5],tolerance_pp=0.1)["selected"]["policy"] == "full-depth"
    assert select_policy(logits,targets,[0.5],tolerance_pp=25)["selected"]["threshold"] == 0.5


@pytest.mark.parametrize("threshold", [0,0.5,0.9,1])
def test_cached_exit_logits_use_first_crossing_and_last_stage_fallback(threshold):
    logits,_ = policy_fixture()
    chosen,exits = exits_from_logits(logits,threshold)
    assert torch.equal(chosen, logits[torch.arange(4),exits-1])
    assert ((exits >= 1) & (exits <= 3)).all()
    if threshold == 1:
        assert exits.tolist() == [3]*4


def test_transition_counts_partition_every_sample_and_distinguish_eventual_from_final():
    predictions = torch.tensor([[1,0,1],[0,1,0],[1,1,1],[0,0,0]])
    logits = torch.nn.functional.one_hot(predictions,2).float()*4
    result = stage_transitions(logits,torch.zeros(4,dtype=torch.long))
    for row in result["transitions"]:
        assert [row[k] for k in ("rescued","damaged","incorrect_to_incorrect","correct_to_correct")] == [1,1,1,1]
    assert result["stage1_correct"] == 2
    assert result["stage1_errors_correct_at_final"] == 0
    assert result["stage1_errors_ever_corrected"] == 1
    assert result["stage1_correct_wrong_at_final"] == 0
    assert result["stage1_correct_ever_broken"] == 1
    assert result["accuracy_by_stage"] == [0.5]*3


def test_validation_split_is_disjoint_complete_and_reproducible():
    source = TensorDataset(torch.rand(32,1,28,28),torch.arange(32)%10,torch.arange(32))
    train,val = validation_split(source,size=8)
    again_train,again_val = validation_split(source,size=8)
    assert len(train) == 24 and len(val) == 8
    assert not set(train.tensors[2].tolist()) & set(val.tensors[2].tolist())
    assert set(train.tensors[2].tolist()) | set(val.tensors[2].tolist()) == set(range(32))
    assert split_record(train) == split_record(again_train)
    assert split_record(val) == split_record(again_val)


def test_resize_is_fixed_and_retains_targets_and_sample_ids():
    source = TensorDataset(torch.linspace(-1,1,2*784).reshape(2,1,28,28),torch.tensor([1,2]),torch.tensor([42,53]))
    resized = resize_direct(source)
    assert resized.tensors[0].shape == (2,1,16,16)
    assert resized.tensors[0].min() >= -1 and resized.tensors[0].max() <= 1
    assert torch.equal(resized.tensors[1],source.tensors[1])
    assert torch.equal(resized.tensors[2],source.tensors[2])


def test_sample_standard_deviation_ddof_one_and_no_fake_missing_value():
    result = summary([1,3])
    assert result["mean"] == 2
    assert result["std"] == pytest.approx(math.sqrt(2))
    assert summary([])["mean"] is None
    assert summary([1])["std"] is None


def test_full_plan_has_unique_jobs_and_exact_five_seed_replications():
    jobs = plan([42,43,44,45,46],["mnist","fashion-mnist"],FAMILIES)
    assert len(jobs) == 190
    assert len({case_id(job) for job in jobs}) == 190
    replication = [j for j in jobs if j["dataset"] == "mnist" and j["family"] == "replication"]
    assert len(replication) == 20
    assert {j["seed"] for j in replication} == {42,43,44,45,46}
