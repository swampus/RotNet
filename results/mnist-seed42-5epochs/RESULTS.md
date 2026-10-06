# RotNet measured results

Run: `mnist-seed42-5epochs`. Status: **completed**.

Dataset: `mnist`; seed: 42; requested epochs: 5; device: `cpu`; CPU threads: 2.

Executed data sizes: 60000 training and 10000 test samples.

| Model | Test accuracy | Trainable / total params | Core params | Core multiply/add ops | Train s | Batched ms/sample | Batch=1 ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 96.39% | 269322 / 269322 | 65792 | 131072.0 | 10.28 | 0.0024 | 0.0461 |
| low-rank | 96.77% | 220170 / 220170 | 16640 | 32736.0 | 9.21 | 0.0057 | 0.1517 |
| rotation | 96.74% | 208650 / 208650 | 5120 | 10240.0 | 32.88 | 0.0222 | 1.0306 |
| adaptive-rotation | 96.76% | 208650 / 208650 | 5120 | 1886.2 | 35.56 | 0.0166 | 0.5250 |

Adaptive accuracy and core operations above use the configured default threshold. Timings are medians of repeated model-only passes; evaluation wall time is recorded separately. See metrics.json for batch sizes, sample counts, individual passes and exclusions.

Rotation accuracy differs from dense by +0.35 percentage points. Its batched inference takes 9.36 times the dense baseline time.

**RotationNet was slower in wall-clock batched inference despite its lower theoretical core arithmetic count.**

## Adaptive threshold sweep

| Threshold | Accuracy | Mean stages | Median stages | Batched ms/sample | Batch=1 ms/sample | Mean stages, correct | Mean stages, incorrect |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 96.81% | 1.151 | 1.0 | 0.0150 | 0.4214 | 1.084 | 3.163 |
| 0.80 | 96.74% | 1.269 | 1.0 | 0.0159 | 0.4582 | 1.157 | 4.607 |
| 0.90 | 96.76% | 1.474 | 1.0 | 0.0166 | 0.5250 | 1.320 | 6.071 |
| 0.95 | 96.76% | 1.716 | 1.0 | 0.0175 | 0.5978 | 1.538 | 7.012 |
| 0.99 | 96.77% | 2.453 | 1.0 | 0.0193 | 0.8445 | 2.277 | 7.737 |

Final head with all stages: accuracy 96.77%; batched inference 0.0221 ms/sample.

This full-depth reference uses the same trained adaptive weights and just the final classifier call. It is separate from RotationNet, which has a different training objective.

| Threshold | Exit counts, stages 1 through last |
|---|---|
| 0.70 | 9593, 121, 59, 42, 30, 22, 19, 114 |
| 0.80 | 9345, 167, 94, 53, 40, 35, 20, 246 |
| 0.90 | 8966, 197, 136, 79, 65, 47, 35, 475 |
| 0.95 | 8475, 278, 198, 111, 89, 44, 60, 745 |
| 0.99 | 6893, 582, 421, 206, 166, 105, 87, 1540 |

## Confidence at the default-threshold exit

| Stage | Samples | Mean confidence | Exit accuracy |
|---|---:|---:|---:|
| 1 | 8966 | 0.990 | 0.995 |
| 2 | 197 | 0.932 | 0.883 |
| 3 | 136 | 0.933 | 0.890 |
| 4 | 79 | 0.928 | 0.873 |
| 5 | 65 | 0.928 | 0.846 |
| 6 | 47 | 0.922 | 0.830 |
| 7 | 35 | 0.917 | 0.886 |
| 8 | 475 | 0.689 | 0.562 |

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
& C:\srdev\work\research\RotNet\.venv\Scripts\python.exe scripts\run_experiment.py --dataset mnist --epochs 5 --device cpu --seed 42 --threads 2 --run-name mnist-seed42-5epochs
```

Configuration, source hashes and environment are saved with every attempt. Executed experiments also save split IDs and raw timing repeats.
