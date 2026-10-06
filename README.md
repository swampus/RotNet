# RotNet

A small Python/PyTorch research experiment asking whether a dense **hidden** transformation can be replaced by structured pairwise rotations, and whether confidence-based early exits offer a useful accuracy/compute trade-off. This is exploratory, makes no novelty claims, and does not assume that the approach works. Negative results are valid.

The tested component is the 256 → 256 hidden transformation. Every model retains a dense 784 → 256 input projection, GELU, and a 256 → 10 classifier. There are no convolutions or augmentation. CPU execution is supported. See [RESULTS.md](RESULTS.md) for execution status and actual measurements; missing dependencies are reported as pending, never as invented results.

## Windows setup and execution

Run these commands in PowerShell. Using the venv executable directly avoids activation-policy issues. Python 3.10 or newer is required; the inspected local interpreter is Python 3.12.

```powershell
Set-Location 'C:\srdev\work\research\RotNet'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The first installation uses CPU PyTorch wheels; the requirements command adds NumPy, matplotlib, and pytest. An internet connection is needed for dependencies and the initial dataset download. Matching torch/torchvision versions are resolved together. Each run saves its installed versions and `requirements-resolved.txt` so the broad dependency ranges can be replaced with a known environment after a successful run.

A tiny real-MNIST pipeline smoke test (all four models, **not** a research result):

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --dataset mnist --epochs 1 --device cpu --seed 42 --train-limit 128 --test-limit 64 --batch-size 64 --eval-batch-size 64 --benchmark-samples 64 --single-sample-benchmark-samples 16 --timing-repeats 1 --run-name smoke-mnist --no-update-summary
```

The main experiment uses all 60,000 official training samples and all 10,000 official test samples, five epochs per model, AdamW, learning rate 0.001, no weight decay, seed 42, two CPU threads, training batch 128, and evaluation batch 256:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --dataset mnist --epochs 5 --device cpu --seed 42 --threads 2 --run-name mnist-seed42-5epochs
```

Optional commands:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --dataset fashion-mnist --epochs 5 --device cpu --seed 42 --run-name fashion-seed42-5epochs
.\.venv\Scripts\python.exe scripts\run_experiment.py --model adaptive-rotation --threshold 0.95 --epochs 5 --device cpu
.\.venv\Scripts\python.exe scripts\run_experiment.py --preflight --no-update-summary
.\.venv\Scripts\python.exe scripts\run_experiment.py --help
```

Run names must be unique; existing directories are never overwritten. Choose another name if an earlier dependency check created that directory. Omitting `--run-name` creates a UTC timestamp name. Use `--no-download` for datasets already cached in `data/`. Without an editable install, always run the entry point from `scripts/`; it locates the repository package itself.

## Models and pairing

| Model | Hidden/core transformation | Core trainable parameters at defaults |
|---|---|---:|
| DenseNet | Linear(256,256), GELU | 65,792 |
| LowRankNet | Linear(256,32,bias=False), Linear(32,256), GELU | 16,640 |
| RotationNet | Eight rotation → diagonal scale → bias → GELU stages | 5,120 |
| AdaptiveRotationNet | Same stages; shared classifier after each stage | 5,120 |

These counts are architecture-derived, **not measured training results**. The input projection contains 200,960 parameters and the shared classifier 2,570. All parameters are initially trainable; buffers are not parameters. The adaptive classifier is counted once, and excluded from the core count even though it is called repeatedly. Low-rank factors have no intermediate nonlinearity.

For power-of-two width `n`, stages use strides `1, 2, 4, ..., n/2`. For a stride `s`, split coordinates into contiguous blocks of size `2*s`; within each block pair offset `j` with offset `j+s`, for `j=0..s-1`. Equivalently, pair `i` with `i XOR s`, taking the member whose `s` bit is zero as the first member. Each coordinate appears exactly once in a stage. For `n=8`:

```text
stride 1: (0,1) (2,3) (4,5) (6,7)
stride 2: (0,2) (1,3) (4,6) (5,7)
stride 4: (0,4) (1,5) (2,6) (3,7)
```

Each pair has its own trainable angle. For inputs `(a,b)`, output is `(a cosθ - b sinθ, a sinθ + b cosθ)`. `PairwiseRotation` reshapes the vector into blocks, performs elementwise operations, stacks the two halves, and reshapes back. It constructs no `n × n` matrix, sparse or dense. Leading batch dimensions are supported. Eight butterfly stages at `n=256` provide paths for information to propagate across the whole vector; this is a connectivity property, not a guarantee that learned angles use those paths effectively.

Angles start uniformly in `[-π/4, π/4]`; scales start at 1 and biases at 0. Scale and bias relax pure rotations' norm preservation. Eight successive GELUs can attenuate signals; this is a methodological concern to investigate if optimization is poor, not a reason to conceal a failed run. Defaults are fixed before test evaluation. All models use identical initial projection/classifier weights; RotationNet and AdaptiveRotationNet also start with identical core weights under the same seed.

## Adaptive training and inference

Training executes all eight stages and applies the same classifier at each stage. The loss is the equal mean of cross-entropy over all samples and stages. There is no hard stopping, gate loss, or test-driven early stopping during training. Ordinary `forward` returns final-stage logits; `forward_all` returns `[batch, stage, class]` logits.

At inference, call `model.eval()` then `model.infer(images, threshold=0.90)`. Confidence is the maximum softmax probability. A sample exits at the first stage whose confidence is **at least** the threshold. Stage 8 is unconditional if none qualifies. Active samples are compacted so later stages do not process samples that already exited. `AdaptiveOutput` contains prediction, 1-based exit stage, confidence and logits **from that exit**, not from a later unexecuted stage.

Every adaptive run evaluates the preset thresholds 0.70, 0.80, 0.90, 0.95, and 0.99; `--thresholds` can add thresholds and `--threshold` selects the default reported in the model comparison. Each threshold is evaluated with actual early stopping, not retrospective savings inferred from executing all stages. A separate reference evaluates the same adaptive weights with all stages and only the final head. The required threshold sweep is descriptive; selecting a deployment threshold would require a validation set and independent test evaluation.

## Arithmetic and timing

For a square dense transformation `y = Wx`, the parameter and arithmetic costs are roughly **O(n²)**. A pairwise stage uses **O(n)** arithmetic. A full butterfly has `log₂(n)` stages, so its mixing cost is **O(n log n)**; a fixed or early-exit prefix of `k` stages costs **O(kn)**.

The accounting counts scalar multiplies and additions, including affine biases (one multiply and one add count as two operations). It explicitly excludes GELU, angle sin/cos, softmax, threshold decisions, indexing, allocations, and memory traffic. At `n=256`:

| Core | Multiply/add operations per sample | Core GELU coordinates per sample |
|---|---:|---:|
| Dense with bias | 131,072 | 256 |
| Low rank, r=32 | 32,736 | 256 |
| Eight rotation stages, including scale/bias | 10,240 | 2,048 |

One rotation stage has 128 pairs × (4 multiplies + 2 additions) = 768 operations; scaling and bias add 512, totaling 1,280. Angles' sin/cos are recomputed per executed stage per batch, not cached or counted as scalar multiply/add operations. These formulas are verified by tests and appear alongside exclusions in `metrics.json`.

The dense input projection alone costs 401,408 multiply/add operations per sample; one classifier call costs 5,120. Adaptive inference calls that classifier once per executed stage. Whole-model arithmetic includes the shared projection and the appropriate number of heads. This makes core-only savings much larger than whole-model savings.

Wall-clock inference uses `time.perf_counter`, warm-up batches, three complete timing passes, and the median plus raw pass times. Data is preloaded outside timing; transfers, loading, CSV and metric aggregation are excluded. Final softmax/prediction extraction is included for every model; adaptive timing additionally includes all gating and batch compaction. CUDA is synchronized at timing boundaries. The benchmark uses the same ordered test prefix (default 1,000 samples) for every model and threshold, at batch size 256 and separately at batch size 1. `--benchmark-samples`, `--single-sample-benchmark-samples`, `--eval-batch-size` and `--timing-repeats` control these costs. Adaptive mean stages on the timed prefix are recorded separately from full-test mean stages. Full-set evaluation wall time, including metric collection, is also recorded and is not used as a compute-only latency comparison.

Training time includes batches, optimization and epoch metric collection, but excludes dataset downloads, model construction, evaluation, saving and plotting. Epoch times are saved. Model order is fixed, so thermal effects and machine background load remain possible confounders.

**Caveats:** theoretical arithmetic counts are not actual CPU speed. Optimized dense BLAS may outperform small custom operations. Fewer parameters do not imply better accuracy, latency, training efficiency or generalization. RotationNet may simply lose accuracy. Confidence checks, extra classifiers, compaction and synchronization may erase adaptive savings. The dense input projection dominates much of the model. One short run and one seed cannot establish a robust advantage.

## Artifacts and difficulty analysis

Each `results/<run>/` contains:

- `config.json`, `environment.json`, `source_hashes.json`, and installed-package `requirements-resolved.txt` when pip is available.
- `metrics.json` with execution status, counts, operation conventions, training history, test accuracy, timing repeats, threshold sweeps, and difficulty summaries.
- `splits.json` with original dataset sample IDs and executed sizes, after data loading succeeds.
- `predictions.csv` with `model,sample_id,target,prediction,correct,confidence,exit_stage,threshold,policy,logit_0,...,logit_9`.
- `checkpoints/<model>.pt` with final weights and configuration; checkpoint loading should instantiate the recorded model configuration first.
- `RESULTS.md` generated from the recorded values, plus five PNG plots for runs including all four models.
- `error.log` if an execution raises an exception.

The root `RESULTS.md` reflects the latest attempt unless `--no-update-summary` is used. A failed or dependency-blocked attempt records its status and any completed models; it does not report unexecuted models as having zero accuracy or create fake plots. A pending run's predictions file is header-only. A dependency preflight can run without importing torch at CLI startup.

For each adaptive threshold, metrics include average and median stages, exit counts across all stages, average stages for correct and incorrect predictions, and confidence/accuracy/count at each exit stage. Empty groups use JSON `null`. The CSV has one adaptive row per sample **per threshold**, plus a full-depth reference row (`policy=full-depth`, blank threshold). Do not combine these into an exit histogram without filtering. `sample_id` always refers to the original official test split. For fixed dense/low-rank models `exit_stage` is blank; fixed RotationNet uses its full stage count.

The five plots show accuracy by model, core parameters by model, adaptive accuracy versus average stages, default-threshold exit counts, and confidence distributions versus exit stage. Accuracy plots show the full 0–100% scale, parameter/count bars begin at zero, all exit stages are visible, and confidence plots use 0–1. The confidence box plot hides individual outlier dots to avoid clutter; the CSV retains every value. Thresholds and the same-model full-depth reference are labeled. Plotting occurs only after real metrics exist.

Confidence is part of the exit rule. Its association with stages is expected and does not establish input difficulty. Correctness is also an imperfect difficulty proxy: confidently wrong predictions can exit early. No statistical interpretation or causal claim is made.

## Reproducibility and tests

Python, NumPy and torch seeds are set; deterministic torch algorithms are enabled, with no silent fallback. Training sample order is reset identically for each model, data-loader workers are zero, and model/optimizer budgets match. Inputs are normalized using the fixed transform `(pixel/255 - 0.5)/0.5` for both datasets. Full official splits are used by default. Limits choose seeded random subsets and retain original sample IDs. There is no hyperparameter search, checkpoint selection or validation tuning in Phase 1. Reproducibility across different PyTorch releases, devices and hardware is not guaranteed; environment and source hashes identify each attempt. See the official [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) and [torchvision MNIST interface](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html).

The tests cover all eight requested properties, pair/angle correspondence, full butterfly connectivity, shared initialization, operation accounting, training gradients at every adaptive stage, chosen-exit logits/sample ordering, invalid configurations, report/export behavior and a synthetic **test-only** pipeline through all five plots. The synthetic fixture is never presented as MNIST evidence.

If runtime dependencies are unavailable, the standard-library infrastructure tests and Python syntax checks can still execute:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_infrastructure.py -v
.\.venv\Scripts\python.exe -m compileall -q rotnet scripts tests
```

These checks do not verify PyTorch numerical behavior and do not substitute for the full pytest suite or real-MNIST smoke/full experiments.

## Layout

```text
rotnet/models/       four models and structured operator
rotnet/data.py       deterministic official splits and tensor loading
rotnet/train.py      seeding and fixed-budget training
rotnet/evaluate.py   true adaptive execution and timing
rotnet/metrics.py    explicit core/whole-model accounting
rotnet/plots.py      five matplotlib outputs
rotnet/reporting.py  CSV/JSON and results reports
rotnet/experiment.py CLI and run orchestration
scripts/            direct entry point
tests/              numerical and infrastructure tests
results/            recorded attempts and successful experiments
```
