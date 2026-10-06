# RotNet measured results

Run: `smoke-mnist`. Status: **completed**.

Dataset: `mnist`; seed: 42; requested epochs: 1; device: `cpu`; CPU threads: 2.

Executed data sizes: 128 training and 64 test samples.

| Model | Test accuracy | Trainable / total params | Core params | Core multiply/add ops | Train s | Batched ms/sample | Batch=1 ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 29.69% | 269322 / 269322 | 65792 | 131072.0 | 0.01 | 0.0034 | 0.0487 |
| low-rank | 31.25% | 220170 / 220170 | 16640 | 32736.0 | 0.00 | 0.0029 | 0.0515 |
| rotation | 20.31% | 208650 / 208650 | 5120 | 10240.0 | 0.01 | 0.0145 | 0.3082 |
| adaptive-rotation | 23.44% | 208650 / 208650 | 5120 | 10240.0 | 0.01 | 0.0233 | 0.7272 |

Adaptive accuracy and core operations above use the configured default threshold. Timings are medians of repeated model-only passes; evaluation wall time is recorded separately. See metrics.json for batch sizes, sample counts, individual passes and exclusions.

Rotation accuracy differs from dense by -9.38 percentage points. Its batched inference takes 4.24 times the dense baseline time.

RotationNet lost accuracy in this run.

**RotationNet was slower in wall-clock batched inference despite its lower theoretical core arithmetic count.**

## Adaptive threshold sweep

| Threshold | Accuracy | Mean stages | Median stages | Batched ms/sample | Batch=1 ms/sample | Mean stages, correct | Mean stages, incorrect |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 23.44% | 8.000 | 8.0 | 0.0235 | 0.7209 | 8.000 | 8.000 |
| 0.80 | 23.44% | 8.000 | 8.0 | 0.0232 | 0.7249 | 8.000 | 8.000 |
| 0.90 | 23.44% | 8.000 | 8.0 | 0.0233 | 0.7272 | 8.000 | 8.000 |
| 0.95 | 23.44% | 8.000 | 8.0 | 0.0230 | 0.8032 | 8.000 | 8.000 |
| 0.99 | 23.44% | 8.000 | 8.0 | 0.0239 | 0.7700 | 8.000 | 8.000 |

Final head with all stages: accuracy 23.44%; batched inference 0.0145 ms/sample.

This full-depth reference uses the same trained adaptive weights and just the final classifier call. It is separate from RotationNet, which has a different training objective.

| Threshold | Exit counts, stages 1 through last |
|---|---|
| 0.70 | 0, 0, 0, 0, 0, 0, 0, 64 |
| 0.80 | 0, 0, 0, 0, 0, 0, 0, 64 |
| 0.90 | 0, 0, 0, 0, 0, 0, 0, 64 |
| 0.95 | 0, 0, 0, 0, 0, 0, 0, 64 |
| 0.99 | 0, 0, 0, 0, 0, 0, 0, 64 |

## Confidence at the default-threshold exit

| Stage | Samples | Mean confidence | Exit accuracy |
|---|---:|---:|---:|
| 1 | 0 | n/a (empty group) | n/a (empty group) |
| 2 | 0 | n/a (empty group) | n/a (empty group) |
| 3 | 0 | n/a (empty group) | n/a (empty group) |
| 4 | 0 | n/a (empty group) | n/a (empty group) |
| 5 | 0 | n/a (empty group) | n/a (empty group) |
| 6 | 0 | n/a (empty group) | n/a (empty group) |
| 7 | 0 | n/a (empty group) | n/a (empty group) |
| 8 | 64 | 0.120 | 0.234 |

No tested adaptive threshold matched the same model's full-depth accuracy while reducing measured batched inference time.

## Methodological limits

- One seed and a short training budget are exploratory, not evidence of a general advantage.
- Dense/low-rank use one core GELU; rotation models use one per stage. Depth and expressivity differ.
- The learned dense input projection remains the largest shared component; core savings are not whole-model savings.
- Multiply/add counts exclude nonlinearities, trigonometry, gating, softmax and memory costs. Dense BLAS may be faster.
- Softmax confidence is uncalibrated and explicitly controls exits. Confidence/stage association is expected and does not establish difficulty.
- Thresholds are preset, evaluated on the same test set, and not selected for deployment using these test results.
- Timing repeats measure runtime variability, not independent training seeds; test-prefix timing may differ from full-set timing.
- No novelty claim, no guaranteed accuracy retention; negative results are valid.

## Reproduction

Command recorded for this run:

```powershell
& C:\srdev\work\research\RotNet\.venv\Scripts\python.exe scripts\run_experiment.py --dataset mnist --epochs 1 --device cpu --seed 42 --train-limit 128 --test-limit 64 --batch-size 64 --eval-batch-size 64 --benchmark-samples 64 --single-sample-benchmark-samples 16 --timing-repeats 1 --run-name smoke-mnist --no-update-summary
```

Configuration, source hashes and environment are saved with every attempt. Executed experiments also save split IDs and raw timing repeats.
