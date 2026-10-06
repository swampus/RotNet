# RotNet measured results

Run: `attempt-mnist-seed42-5epochs`. Status: **blocked**.

Runtime results are pending. This run did not complete; missing values are not zeroes.

Reason: Dependencies unavailable: torch: ModuleNotFoundError: No module named 'torch'; torchvision: ModuleNotFoundError: No module named 'torchvision'; numpy: ModuleNotFoundError: No module named 'numpy'; matplotlib: ModuleNotFoundError: No module named 'matplotlib'

Dataset: `mnist`; seed: 42; requested epochs: 5; device: `cpu`; CPU threads: 2.


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
& C:\srdev\work\research\RotNet\.venv\Scripts\python.exe scripts\run_experiment.py --dataset mnist --epochs 5 --device cpu --seed 42 --threads 2 --run-name attempt-mnist-seed42-5epochs
```

Configuration, source hashes and environment are saved with every attempt. Executed experiments also save split IDs and raw timing repeats.
