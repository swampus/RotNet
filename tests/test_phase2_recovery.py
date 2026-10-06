import json

import pytest

from rotnet.phase2.report import completed
from rotnet.phase2.runner import attempt_history


def test_attempt_history_preserves_failed_and_interrupted_attempts():
    earlier = {"status":"interrupted", "path":"attempt001", "error":"worker lost"}
    previous = {"status":"failed", "path":"attempt002", "error":"numerical mismatch",
                "attempt":2, "previous_attempts":[earlier]}
    history = attempt_history(previous)
    assert history[0] == earlier
    assert history[1]["status"] == "failed"
    assert history[1]["path"] == "attempt002"
    assert history[1]["error"] == "numerical mismatch"
    assert previous["previous_attempts"] == [earlier]
    assert attempt_history(None) == []


def test_completed_filter_does_not_treat_failed_or_blocked_cases_as_zero_accuracy():
    suite = {"cases": {
        "a":{"status":"completed","job":{"dataset":"mnist","family":"replication"}},
        "b":{"status":"failed","job":{"dataset":"mnist","family":"replication"}},
        "c":{"status":"blocked","job":{"dataset":"fashion-mnist","family":"replication"}},
    }}
    assert len(completed(suite,"mnist","replication")) == 1
    assert completed(suite,"fashion-mnist","replication") == []


@pytest.mark.parametrize("option,value",[("--tolerance-pp","0.2"),("--validation-size","5001")])
def test_resume_cannot_change_validation_selection_protocol(monkeypatch,tmp_path,option,value):
    import rotnet.phase2.runner as runner
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    run_dir=tmp_path/"results"/"phase2"/"fixture"
    run_dir.mkdir(parents=True)
    suite={"settings":{"smoke":False,"seeds":[42],"split_seed":202602,
                       "validation_size":5000,"tolerance_pp":0.1}}
    (run_dir/"suite.json").write_text(json.dumps(suite))
    with pytest.raises(ValueError,match="differs from recorded"):
        runner.main(["--run-dir",str(run_dir),"--resume","--seeds","42",option,value])


def test_resume_rejects_changed_model_mathematics(monkeypatch,tmp_path):
    import rotnet.phase2.runner as runner
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    monkeypatch.setattr(runner,"source_hashes",lambda:{"rotnet/phase2/models.py":"new"})
    run_dir=tmp_path/"results"/"phase2"/"fixture"
    run_dir.mkdir(parents=True)
    suite={"settings":{"smoke":False,"seeds":[42],"split_seed":202602,
                       "validation_size":5000,"tolerance_pp":0.1},
           "source_sha256":{"rotnet/phase2/models.py":"old"}}
    (run_dir/"suite.json").write_text(json.dumps(suite))
    with pytest.raises(ValueError,match="training/evaluation source changed"):
        runner.main(["--run-dir",str(run_dir),"--resume","--seeds","42"])


def test_resume_marks_worker_running_before_reporting(monkeypatch,tmp_path):
    import rotnet.phase2.runner as runner
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    monkeypatch.setattr(runner,"source_hashes",lambda:{})
    monkeypatch.setattr(runner.torch,"set_num_threads",lambda count:None)
    monkeypatch.setattr(runner.torch,"set_num_interop_threads",lambda count:None)
    monkeypatch.setattr(runner,"verify_manifest",lambda manifest:{"status":"passed"})
    def unavailable(*args,**kwargs):
        raise RuntimeError("test fixture: no cached dataset")
    monkeypatch.setattr(runner,"load_data",unavailable)
    observed=[]
    monkeypatch.setattr(runner,"persist",lambda suite,args:observed.append(suite["status"]))
    run_dir=tmp_path/"results"/"phase2"/"fixture"
    run_dir.mkdir(parents=True)
    suite={"status":"interrupted","settings":{"smoke":False,"seeds":[42],"split_seed":202602,
                       "validation_size":5000,"tolerance_pp":0.1},
           "source_sha256":{},"config":{"threads":2},"environment":{},
           "phase1_manifest":{},"dataset_failures":{},"cases":{}}
    (run_dir/"suite.json").write_text(json.dumps(suite))
    runner.main(["--run-dir",str(run_dir),"--resume","--seeds","42","--datasets","mnist"])
    assert observed[0] == "running"
    assert observed[-1] == "completed-with-blocked-dataset"
