# Phase 2 RotNet: replication and falsification

## 1. Executive summary

Status: **completed-with-blocked-dataset**. Run directory: `results\phase2\mnist-five-seeds`.

Completed cases: 95; failed cases: 0; blocked cases: 95.

Accuracy, training/inference times and exits are measured. Parameter counts are enumerated; scalar multiply/add arithmetic is theoretical accounting. SD is the sample standard deviation across seeds (ddof=1), not a confidence interval.

Seeds: 42, 43, 44, 45, 46. Each model trains for 5 epochs with unchanged Phase 1 AdamW settings, batches 128/256, width 256, rank 32, two CPU threads.

Across 5 paired MNIST seeds, RotationNet minus Dense accuracy averaged -0.386 pp, range -1.460 to +0.350 pp. This is descriptive consistency evidence, not a significance or noninferiority test. In the separate 16x16/no-projection comparison, Dense averaged 96.810% and Rotation 93.838%; the structured replacement lost 2.972 pp.
Learned minus identity mean accuracy is +0.324 pp. Learned minus frozen-random is -0.078 pp. These controls, including coordinate-only and no-affine variants, test necessity; they do not uniquely identify a causal mechanism.
Validation-selected policies use mean 1.024/8 test stages. Mean test accuracy difference from the same weights' final head is -0.040 pp (range -0.230 to +0.120); mean whole-model multiply/add savings are 2.11%. A validation tolerance does not guarantee test accuracy retention. In the separate direct-input preset-0.90 experiment, adaptive whole-model arithmetic was 24.83% HIGHER than fixed Rotation, due to repeated classifier calls, with lower mean accuracy. Fewer stages did not imply cheaper whole-model execution.
Optimized RotationNet takes 3.40× Dense's batch=256 time on this CPU. A dense latency advantage remains. The cached adaptive attempt was 3.05% slower in batched inference.

**Fashion-MNIST: BLOCKED.** No local dataset; network sockets are unavailable. No Fashion-MNIST accuracy or timing is fabricated.

## 2. Multi-seed MNIST replication

| Model/control | Seeds | Mean accuracy ± sample SD (%) | Min–max (%) | Mean core ops | Mean core trainable/total params | Mean train s | Mean batch=256 ms/sample | Mean batch=1 ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive-rotation | 5 | 96.834 ± 0.125 | 96.66–96.98 | 1882.4 | 5120/5120 | 39.49 | 0.01852 | 0.58074 |
| dense | 5 | 97.242 ± 0.530 | 96.39–97.84 | 131072.0 | 65792/65792 | 12.18 | 0.00686 | 0.15624 |
| low-rank | 5 | 97.002 ± 0.248 | 96.77–97.35 | 32736.0 | 16640/16640 | 11.41 | 0.00599 | 0.16720 |
| rotation | 5 | 96.856 ± 0.300 | 96.38–97.10 | 10240.0 | 5120/5120 | 36.56 | 0.02394 | 1.11163 |

Adaptive replication uses the unchanged preset 0.90 threshold and full 60,000-example training split. It is kept separate from validation-selected policies.

## 3. Multi-seed Fashion-MNIST replication

**BLOCKED** — RuntimeError: Dataset not found. You can use download=True to download it

The same suite supports Fashion-MNIST when the official dataset is cached; no dataset-specific tuning is implemented.

## 4. Rotation stage-count ablation

| Model/control | Seeds | Mean accuracy ± sample SD (%) | Min–max (%) | Mean core ops | Mean core trainable/total params | Mean train s | Mean batch=256 ms/sample | Mean batch=1 ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 96.990 ± 0.168 | 96.81–97.22 | 1280.0 | 640/640 | 12.09 | 0.00645 | 0.23911 |
| 2 | 5 | 97.096 ± 0.181 | 96.85–97.36 | 2560.0 | 1280/1280 | 18.37 | 0.01243 | 0.36896 |
| 4 | 5 | 96.986 ± 0.447 | 96.51–97.70 | 5120.0 | 2560/2560 | 26.53 | 0.01870 | 0.61555 |
| 8 | 5 | 96.856 ± 0.300 | 96.38–97.10 | 10240.0 | 5120/5120 | 36.56 | 0.02394 | 1.11163 |

Stages are deterministic prefixes (strides 1,2,...). Fewer stages change connectivity as well as nonlinear depth. Eight-stage values reuse freshly measured Phase 2 replication cases.

Already-trained adaptive classifier at each fixed stage (no threshold):

| Stage | Mean accuracy ± SD (%) |
|---|---:|
| 1 | 96.782 ± 0.257 |
| 2 | 96.866 ± 0.167 |
| 3 | 96.906 ± 0.108 |
| 4 | 96.898 ± 0.112 |
| 5 | 96.880 ± 0.176 |
| 6 | 96.900 ± 0.199 |
| 7 | 96.914 ± 0.174 |
| 8 | 96.782 ± 0.126 |

## 5. Rotation-component controls

| Model/control | Seeds | Mean accuracy ± sample SD (%) | Min–max (%) | Mean core ops | Mean core trainable/total params | Mean train s | Mean batch=256 ms/sample | Mean batch=1 ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| learned | 5 | 96.856 ± 0.300 | 96.38–97.10 | 10240.0 | 5120/5120 | 36.56 | 0.02394 | 1.11163 |
| identity | 5 | 96.532 ± 0.270 | 96.11–96.81 | 10240.0 | 4096/5120 | 30.34 | 0.02381 | 1.11145 |
| random-fixed | 5 | 96.934 ± 0.396 | 96.53–97.46 | 10240.0 | 4096/5120 | 30.41 | 0.02397 | 1.10866 |
| final-gelu | 5 | 96.876 ± 0.276 | 96.48–97.16 | 10240.0 | 5120/5120 | 33.77 | 0.02117 | 0.88911 |
| coordinate-only | 5 | 96.532 ± 0.270 | 96.11–96.81 | 4096.0 | 4096/4096 | 17.94 | 0.00920 | 0.45454 |
| no-affine | 5 | 97.124 ± 0.253 | 96.76–97.40 | 6144.0 | 1024/1024 | 31.28 | 0.02202 | 1.01487 |
| projection-only | 5 | 97.094 ± 0.236 | 96.93–97.51 | 0.0 | 0/0 | 9.55 | 0.00492 | 0.10465 |

Identity retains theta=0 as frozen parameters and executes the same rotation arithmetic. Coordinate-only removes mixing entirely; it is the same eight-stage coordinate-affine/GELU depth. Random-fixed freezes the original random angles. Final-GELU uses one GELU after all eight affine rotation stages. No-affine removes scale/bias entirely. Projection-only removes the hidden core, retaining input projection/GELU and classifier. Frozen parameters are included in total but excluded from trainable counts.

## 6. Proper validation-selected early exit

A fixed seeded 55,000/5,000 partition of official training data is shared across all five seeds. Only validation chooses the policy: minimum mean stages subject to accuracy within 0.1 pp of the same weights' final head. The policy JSON is saved before evaluating the official test labels. Full-depth/no-gating is a fallback. This separate experiment has fewer training updates than the 60,000-example replication; it is not pooled with that table.

| Seed | Selected threshold/policy | Validation accuracy (%) | Full validation (%) | Test accuracy (%) | Full test (%) | Mean/median test stages | Core / whole ops | Batch256 / batch1 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.55 | 97.26 | 97.34 | 96.83 | 96.89 | 1.045/1.0 | 1337.6/408096.0 | 0.01248/0.43344 |
| 43 | 0.35 | 97.04 | 97.14 | 97.14 | 97.05 | 1.002/1.0 | 1282.3/407819.5 | 0.00715/0.42688 |
| 44 | 0.1 | 96.92 | 97.00 | 97.05 | 96.93 | 1.000/1.0 | 1280.0/407808.0 | 0.00707/0.43178 |
| 45 | 0.1 | 96.68 | 96.58 | 96.30 | 96.53 | 1.000/1.0 | 1280.0/407808.0 | 0.00776/0.42607 |
| 46 | 0.6 | 97.14 | 97.24 | 97.14 | 97.26 | 1.071/1.0 | 1370.5/408260.5 | 0.01613/0.46430 |

Exit counts for each frozen policy (stages 1..8):

- Seed 42: 9817, 79, 45, 16, 15, 8, 7, 13
- Seed 43: 9993, 4, 0, 1, 1, 0, 0, 1
- Seed 44: 10000, 0, 0, 0, 0, 0, 0, 0
- Seed 45: 10000, 0, 0, 0, 0, 0, 0, 0
- Seed 46: 9753, 104, 44, 18, 21, 17, 9, 34

**The validation tolerance failed to transfer on 2/5 test seeds.**
Seed 45 lost 0.23 pp versus its own full-depth test accuracy; the selected policy was not changed afterward.
Seed 46 lost 0.12 pp versus its own full-depth test accuracy; the selected policy was not changed afterward.

## 7. Stage rescue/damage analysis

Counts below are means across seeds on the same 10,000 official test images; these are not independent new datasets.

| Transition | Rescued mean ± SD | Damaged mean ± SD | Incorrect→incorrect mean | Correct→correct mean |
|---|---:|---:|---:|---:|
| 1→2 | 46.4 ± 8.6 | 38.0 ± 7.5 | 275.4 | 9640.2 |
| 2→3 | 36.8 ± 11.1 | 32.8 ± 8.2 | 276.6 | 9653.8 |
| 3→4 | 29.8 ± 5.3 | 30.6 ± 5.8 | 279.6 | 9660.0 |
| 4→5 | 31.2 ± 6.0 | 33.0 ± 4.8 | 279.0 | 9656.8 |
| 5→6 | 32.2 ± 10.3 | 30.2 ± 7.6 | 279.8 | 9657.8 |
| 6→7 | 32.0 ± 7.8 | 30.6 ± 4.9 | 278.0 | 9659.4 |
| 7→8 | 23.2 ± 5.3 | 36.4 ± 15.5 | 285.4 | 9655.0 |

| Seed | Stage1 correct | Stage1 errors eventually correct at final | Errors ever corrected | Stage1 correct broken at final | Correct ever broken |
|---|---:|---:|---:|---:|---:|
| 42 | 9657 | 80 | 113 | 60 | 111 |
| 43 | 9665 | 78 | 125 | 82 | 135 |
| 44 | 9661 | 81 | 113 | 58 | 119 |
| 45 | 9690 | 75 | 117 | 70 | 110 |
| 46 | 9718 | 63 | 106 | 107 | 170 |

## 8. 16x16 MNIST: no dense input projection

| Model/control | Seeds | Mean accuracy ± sample SD (%) | Min–max (%) | Mean core ops | Mean core trainable/total params | Mean train s | Mean batch=256 ms/sample | Mean batch=1 ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive-rotation | 5 | 93.236 ± 0.307 | 92.71–93.49 | 3834.7 | 5120/5120 | 34.23 | 0.02019 | 1.06504 |
| dense | 5 | 96.810 ± 0.159 | 96.56–97.00 | 131072.0 | 65792/65792 | 7.03 | 0.00243 | 0.10236 |
| low-rank | 5 | 96.100 ± 0.143 | 95.91–96.31 | 32736.0 | 16640/16640 | 6.57 | 0.00130 | 0.11209 |
| rotation | 5 | 93.838 ± 0.448 | 93.17–94.24 | 10240.0 | 5120/5120 | 31.01 | 0.01890 | 1.05231 |

| Model | Mean core ops | Mean whole-model ops |
|---|---:|---:|
| adaptive-rotation | 3834.7 | 19173.4 |
| dense | 131072.0 | 136192.0 |
| low-rank | 32736.0 | 37856.0 |
| rotation | 10240.0 | 15360.0 |

**Direct-input adaptive execution used 24.83% more whole-model multiply/add arithmetic than fixed Rotation**, despite fewer stages, and achieved lower mean accuracy. The classifier is four times the arithmetic cost of one rotation/affine stage.

Fixed bilinear antialiased resizing; flatten directly to 256. No input projection or input GELU remains. All models share this representation. Adaptive uses the preset 0.90 policy. These results are separate from 28x28; resolution and model capacity both change.

## 9. Runtime optimization

Inference-only copies cache sin/cos and reuse adaptive exit confidences. Learned mathematics and original weights are preserved. All-stage logits on every official test image must match at atol=1e-6, rtol=1e-5; predictions/exits must match exactly. Five implementations are measured in seeded shuffled order within each timing repetition, at batch 1 and 256. Compilation is optional and not attempted; this comparison measures the portable eager cache path.

| Implementation | Seeds | Mean batch256 ms/sample ± SD | Mean batch1 ms/sample ± SD |
|---|---:|---:|---:|
| dense | 5 | 0.00691 ± 0.00044 | 0.15590 ± 0.00118 |
| original-rotation | 5 | 0.02405 ± 0.00071 | 1.10837 ± 0.00266 |
| original-adaptive | 5 | 0.01814 ± 0.00022 | 0.58251 ± 0.01405 |
| optimized-rotation | 5 | 0.02348 ± 0.00059 | 0.94873 ± 0.00419 |
| optimized-adaptive | 5 | 0.01869 ± 0.00033 | 0.60405 ± 0.01455 |

Maximum observed absolute logit error across verified test passes: 0. Exact exit and predicted-class agreement passed for completed runtime cases. Seed-42 CPU operator profiles (including calls/time for trig, GELU, indexing and matrix multiplies) are saved per batch size.

Cached rotation was 14.40% faster at batch=1. This optimization attempt is retained regardless of the outcome.

Cached adaptive was 3.70% slower at batch=1. This optimization attempt is retained regardless of the outcome.

Profiling diagnosis: fixed Rotation still makes hundreds of elementwise calls, repeated stacks/concatenations and nine GELUs including the input activation. Caching removes sin/cos but leaves those costs. The adaptive cache path adds indexing/allocation to save selected confidences; seed-42 profiles show 480 versus 400 index calls at batch=256 and 60 versus 50 at batch=1. These instrumented ten-forward profiles diagnose overhead; they are not the latency benchmark itself.

## 10. Failures and negative findings

- fashion-mnist: BLOCKED. local dataset unavailable; network download is unavailable in this session. RuntimeError: Dataset not found. You can use download=True to download it
- Earlier INTERRUPTED attempt `cases/mnist__replication__45__adaptive-rotation__8__learned/attempt001`: Daemon restart terminated the worker; no Python worker remained. Partial attempt preserved; restart required.. Its artifacts remain; only the completed replacement is included in aggregates.
- The initial default pytest invocation hit Windows temporary/cache directory permissions (52 passed, one setup error). A new workspace basetemp and disabled cache provider resolved it; all original tests and the final expanded suite passed.
- A daemon restart interrupted one adaptive seed-45 attempt. The partial attempt was retained, completed cases were skipped, and the replacement restarted the deterministic five-epoch budget.
- Recovery audit found a resumed worker could retain an interim interrupted reporting status. The runner now resets it to running and a regression test covers it; model/training mathematics were unchanged.

Evidence-based answers to the seven research questions:

1. Across 5 paired MNIST seeds, RotationNet minus Dense accuracy averaged -0.386 pp, range -1.460 to +0.350 pp. This is descriptive consistency evidence, not a significance or noninferiority test. In the separate 16x16/no-projection comparison, Dense averaged 96.810% and Rotation 93.838%; the structured replacement lost 2.972 pp.
2. Unknown: Fashion-MNIST is BLOCKED by unavailable local data/network. Generalization beyond MNIST has not been tested.
3. Learned minus identity mean accuracy is +0.324 pp. Learned minus frozen-random is -0.078 pp. These controls, including coordinate-only and no-affine variants, test necessity; they do not uniquely identify a causal mechanism.
4. Per seed, mean 75.4 stage-1 errors are correct at the final stage, while mean 75.4 stage-1 correct predictions are wrong at the final stage. Error rescue is observed; 'difficulty' is not independently measured.
5. Validation-selected policies use mean 1.024/8 test stages. Mean test accuracy difference from the same weights' final head is -0.040 pp (range -0.230 to +0.120); mean whole-model multiply/add savings are 2.11%. A validation tolerance does not guarantee test accuracy retention. In the separate direct-input preset-0.90 experiment, adaptive whole-model arithmetic was 24.83% HIGHER than fixed Rotation, due to repeated classifier calls, with lower mean accuracy. Fewer stages did not imply cheaper whole-model execution.
6. Optimized RotationNet takes 3.40× Dense's batch=256 time on this CPU. A dense latency advantage remains. The cached adaptive attempt was 3.05% slower in batched inference.
7. Only a targeted follow-up is justified: test whether the mixing is necessary after the input-projection control, then assess a fused structured kernel against optimized dense/low-rank baselines and include a harder dataset. The current prototype is not evidence of a practical CPU speed advantage or a general architectural benefit.

## 11. Methodological limitations

- Five seeds use the same official test set. Sample SD reflects initialization/training variation, not independent dataset uncertainty.
- No significance test, equivalence claim, or novelty claim. Small mean differences should not be treated as established superiority.
- Fixed five-epoch budgets may disadvantage deep stacks; tuning each model is deliberately excluded here.
- Replication has 60,000 training images; validation experiments have 55,000. Their outcomes are not pooled.
- Validation searches a finite preset threshold grid; it can overfit validation and does not guarantee the test tolerance.
- Confidence is uncalibrated; later error rescue is descriptive, not an independent definition of difficulty.
- Arithmetic excludes nonlinearities, trigonometry, allocation, softmax, gating and memory. Whole-model counts include repeated classifiers.
- The identity control has redundant zero-angle arithmetic; coordinate-only removes it and supplies the true no-mixing depth control.
- 16x16 changes resolution as well as removing the learned projection; it is a separate controlled comparison, not a direct 28x28 accuracy claim.
- Timing includes final output selection and adaptive gating; it excludes loading, transfers, CSV and optimizer work. Default timed prefix is 1,000 fixed test samples.
- CPU cache/thermal/background-load variation and kernel/library specificity remain; timer repeats are not independent training seeds.
- Interrupted attempts are retained but excluded from accuracy/timing aggregates; completed replacement runs restart the same deterministic five-epoch budget.
- Fashion-MNIST is unexecuted when marked blocked. No conclusion beyond MNIST is supported.

## 12. Recommended Phase 3

First run this unchanged protocol on cached Fashion-MNIST and one harder dataset. Predefine an accuracy-retention margin and a meaningful latency target. Prioritize comparisons against projection-only, coordinate-depth and low-rank controls. If mixing remains useful without a dense input projection, implement a genuinely fused CPU/GPU structured kernel, verify numerical/exit equivalence, and compare both accuracy and end-to-end latency over more hardware and training budgets. If no-mixing controls match RotNet or direct-input accuracy collapses, narrow or stop the architectural claim.

## Reproduction and preservation

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .phase2-tests-reproduction
.\.venv\Scripts\python.exe scripts\run_phase2.py --run-dir results\phase2\reproduction-five-seeds
```

Use a new run directory for fresh training. To resume the measured run and skip completed cases:

```powershell
.\.venv\Scripts\python.exe scripts\run_phase2.py --run-dir "results\phase2\mnist-five-seeds" --resume
```

Completed cases are skipped on resume. Failures retain original attempts; --retry-failed creates a new numbered attempt. Phase 1 results and source files are protected by a before/after SHA-256 manifest.

Phase 1 preservation: {'status': 'passed', 'files_checked': 74, 'changed': []}.

Final tests: **98 passed**, including all 53 original tests. Test command and XML evidence are saved in validation.json/test-results.xml.

Resume source snapshots are saved. Resumes reject changes to model mathematics, training, data or evaluation code; runner/reporting recovery changes are recorded separately from the original source snapshot.
