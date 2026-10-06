# Running Phase 2

Phase 2 is an independent falsification suite in `rotnet/phase2/`. All Phase 1 model, data, training, evaluation, test, documentation, and completed-result files are left unchanged. Every suite snapshots their SHA-256 hashes and verifies them at the end. Nothing writes to the original `RESULTS.md` or completed `results/mnist-seed42-5epochs/` directory.

The suite inherits epochs, optimizer settings, batches, width, rank, thread count, threshold and timing budgets from the completed Phase 1 configuration. It runs seeds **42,43,44,45,46**. Original replication uses all 60,000 official MNIST training images. Ablations and direct-input experiments also use that fixed training budget. A separate adaptive experiment uses 55,000 training / 5,000 validation / 10,000 official test images. The held-out split is fixed with split seed 202602 across model seeds.

## Windows commands

The existing `.venv` now has dependencies installed. From PowerShell:

```powershell
Set-Location 'C:\srdev\work\research\RotNet'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .phase2-tests-reproduction
.\.venv\Scripts\python.exe scripts\run_phase2.py --run-dir results\phase2\my-new-phase2-run
```

The destination must be a **new** descendant of `results/phase2/`. No existing run is overwritten. The suite intentionally loads cached data without attempting network downloads. Missing Fashion-MNIST is recorded as BLOCKED; all available MNIST work continues.

Resume completed/partially completed work without retraining completed cases:

```powershell
.\.venv\Scripts\python.exe scripts\run_phase2.py --run-dir results\phase2\mnist-five-seeds --resume
```

After Fashion-MNIST becomes available, resume the same suite with only that dataset:

```powershell
.\.venv\Scripts\python.exe -c "from torchvision.datasets import FashionMNIST; FashionMNIST('data', train=True, download=True); FashionMNIST('data', train=False, download=True)"
.\.venv\Scripts\python.exe scripts\run_phase2.py --run-dir results\phase2\mnist-five-seeds --resume --datasets fashion-mnist
```

The download command needs unrestricted network access; it is not part of the measured experiment. If using a different cached-data root, pass `--data-dir <path>`. Same-suite resume retains MNIST records. `--retry-failed` creates another numbered attempt for failed cases; their original metrics and error logs remain. Interrupted cases also start a new attempt. Failed runs are not silently included in a five-seed mean.

To run just a family, use `--families replication stages components validation direct16 runtime`. The runtime family requires replication checkpoints in that same suite. A test-only smoke run is isolated from research results:

```powershell
.\.venv\Scripts\python.exe scripts\run_phase2.py --smoke --seeds 42 --datasets mnist --run-dir results\phase2\my-new-smoke
```

Smoke uses 128 training / 64 test images and one epoch, and does not update root `PHASE2_RESULTS.md`. Its own report is labeled test-only.

## Controlled experiments

1. Replication trains each unchanged Phase 1 model afresh at five seeds on each available dataset. Seed 42 is a deterministic replay, not an additional new random initialization. Adaptive replication retains the preset 0.90 threshold. Mean, sample SD (`ddof=1`), min/max, training time, core/whole operations, and both latency regimes are exported.
2. Stage counts 1,2,4 use exactly the corresponding initialized prefix of the eight-stage model; eight-stage results reuse Phase 2 replication. Shorter prefixes change connectivity as well as depth. There are no extra tuning epochs.
3. Component controls retain the same projection and classifier initialization. Identity sets all theta to zero and freezes them; random-fixed freezes the usual randomly initialized theta. Scale/bias remain trainable. Final-GELU removes all intermediate GELUs and applies one after the complete structured affine operator. Coordinate-only removes rotations entirely, retaining eight affine/GELU stages. No-affine removes scales and biases entirely. Projection-only removes the hidden core, testing how much the large learned projection alone explains. All controls are reported, including failures.
4. Validation threshold selection minimizes mean validation stages subject to accuracy no more than **0.1 percentage points** below the same trained model's final head. Candidate thresholds are preset: 0.10..0.95 in 0.05 increments, plus 0.975,0.99,0.995,0.999,1.0. Full depth with no gate is a guaranteed fallback. Ties favor higher validation accuracy then lower threshold. The selected policy is saved before official test evaluation. Only the frozen policy is assessed on test labels; stage diagnostics and label-free timing repeats do not select or revise it. No test-threshold search is performed in this family.
5. Every adaptive checkpoint exports all eight official-test stage logits and predictions. Transition counts partition samples into rescued, damaged, wrong→wrong and correct→correct. Both “ever corrected/broken later” and “correct/wrong at final stage” are recorded separately; transient correction is not counted as final rescue.
6. The 16x16 experiment uses fixed bilinear resizing with antialiasing and direct flattening to 256. It removes both the dense input projection and its GELU. It keeps each core and classifier unchanged, with the same training settings. These results are reported separately; they cannot isolate projection removal from the resolution/capacity change when compared directly with 28x28 accuracy.
7. Runtime copies cache theta sin/cos as reusable buffers and reuse adaptive exit confidence/predictions instead of applying a final softmax twice. They are inference-only, do not modify original weights, and reject training/gradient paths. Verification compares all stages on the full official test set at `atol=1e-6, rtol=1e-5`, with **exact** predictions and exit stages. The five implementations (Dense, original/cached Rotation, original/cached Adaptive) are timed in seeded shuffled orders within repetitions. Batch sizes are 1 and 256; sample prefix, thread count and output formation are identical. Trig caching does not fuse the many remaining elementwise operations. Optional `torch.compile` is not exercised in this suite. CPU operator profiles for seed 42 are separate from timing. PyTorch documents that mathematically identical floating-point computation can differ across execution paths; see its [numerical-accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).

## Artifacts

`PHASE2_RESULTS.md` is regenerated from machine-readable values after each case. While running, tables identify the number of completed seeds; never treat a partial table as a completed five-seed replication. Final reports include all twelve requested sections and seven evidence-based answers.

Under the suite directory:

- `suite.json`: configuration, exact environment/build, settings, source hashes, job statuses and per-case metrics.
- `phase1_manifest.json`: before-state hashes of the untouched Phase 1 source/tests/artifacts; final preservation check is in `suite.json`.
- `requirements-resolved.txt`, `aggregates.json`, `cases.csv`, `evidence_answers.json`, and the suite-local `PHASE2_RESULTS.md`.
- `cases/<case-id>/attemptNNN/`: job, metrics, training history, split IDs, checkpoint and prediction CSV. Validation cases also store selection candidates/policy and validation logits. Adaptive cases store all-stage test logits, stage prediction CSV and transition analysis.
- Runtime cases store full-set equivalence checks, all raw timing passes/counterbalanced orders and operator profiles.
- Final plots display mean ± seed SD, accuracy/operations/latency trade-offs, per-stage accuracy, rescue/damage counts, frozen-policy exits and validation trade-offs, and direct-input whole-model compute. Accuracy scales run from 0 to 100%; bars start at zero. CSV/JSON preserve small differences hidden by a full-scale plot.

All arithmetic counts exclude GELU, trig, softmax, indexing, allocation and memory. Identity's redundant zero-angle math is counted because it is actually executed; the coordinate-only control removes that work. Frozen angles count toward total parameters, not trainable parameters. Whole-model counts include the actual number of classifier calls, so early exits do not automatically imply end-to-end savings when the input projection remains.

No novelty, significance, equivalence or generalization claim is made from five seeds. Each seed uses the same test set, timing is hardware/library dependent, short-budget deep optimization can fail, and a validation constraint can fail to transfer to test. Fashion-MNIST remains unmeasured whenever marked BLOCKED.
