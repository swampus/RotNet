# Phase 2 RotNet: replication and falsification

## 1. Executive summary

Status: **completed**. Run directory: `C:\srdev\work\research\RotNet\results\phase2\smoke-falsification`.

Completed cases: 19; failed cases: 0; blocked cases: 0.

**TEST-ONLY SMOKE RUN: excluded from research conclusions.**

Seeds: 42. Each model trains for 1 epochs with unchanged Phase 1 AdamW settings, batches 128/256, width 256, rank 32, two CPU threads.

## 2. Multi-seed MNIST replication

| Model/control | Seeds | Mean accuracy ± sample SD (%) | Min–max (%) | Mean core ops | Mean core trainable/total params | Mean train s | Mean batch=256 ms/sample | Mean batch=1 ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive-rotation | 1 | 23.438 ± pending SD | 23.44–23.44 | 10240.0 | 5120/5120 | 0.01 | 0.06582 | 0.77503 |
| dense | 1 | 20.312 ± pending SD | 20.31–20.31 | 131072.0 | 65792/65792 | 0.01 | 0.00670 | 0.06131 |
| low-rank | 1 | 20.312 ± pending SD | 20.31–20.31 | 32736.0 | 16640/16640 | 0.00 | 0.00598 | 0.04714 |
| rotation | 1 | 21.875 ± pending SD | 21.88–21.88 | 10240.0 | 5120/5120 | 0.01 | 0.03065 | 0.32729 |

Adaptive replication uses the unchanged preset 0.90 threshold and full 60,000-example training split. It is kept separate from validation-selected policies.

## 3. Multi-seed Fashion-MNIST replication

No completed measurements in this group.

## 4. Rotation stage-count ablation

| Model/control | Seeds | Mean accuracy ± sample SD (%) | Min–max (%) | Mean core ops | Mean core trainable/total params | Mean train s | Mean batch=256 ms/sample | Mean batch=1 ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 31.250 ± pending SD | 31.25–31.25 | 1280.0 | 640/640 | 0.01 | 0.02126 | 0.24001 |
| 2 | 1 | 32.812 ± pending SD | 32.81–32.81 | 2560.0 | 1280/1280 | 0.01 | 0.03251 | 0.33376 |
| 4 | 1 | 28.125 ± pending SD | 28.12–28.12 | 5120.0 | 2560/2560 | 0.01 | 0.07195 | 0.55882 |
| 8 | 1 | 21.875 ± pending SD | 21.88–21.88 | 10240.0 | 5120/5120 | 0.01 | 0.03065 | 0.32729 |

Stages are deterministic prefixes (strides 1,2,...). Fewer stages change connectivity as well as nonlinear depth. Eight-stage values reuse freshly measured Phase 2 replication cases.

Already-trained adaptive classifier at each fixed stage (no threshold):

| Stage | Mean accuracy ± SD (%) |
|---|---:|
| 1 | 31.250 ± pending SD |
| 2 | 29.688 ± pending SD |
| 3 | 29.688 ± pending SD |
| 4 | 25.000 ± pending SD |
| 5 | 28.125 ± pending SD |
| 6 | 21.875 ± pending SD |
| 7 | 23.438 ± pending SD |
| 8 | 23.438 ± pending SD |

## 5. Rotation-component controls

| Model/control | Seeds | Mean accuracy ± sample SD (%) | Min–max (%) | Mean core ops | Mean core trainable/total params | Mean train s | Mean batch=256 ms/sample | Mean batch=1 ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| learned | 1 | 21.875 ± pending SD | 21.88–21.88 | 10240.0 | 5120/5120 | 0.01 | 0.03065 | 0.32729 |
| identity | 1 | 29.688 ± pending SD | 29.69–29.69 | 10240.0 | 4096/5120 | 0.01 | 0.08493 | 1.04901 |
| random-fixed | 1 | 23.438 ± pending SD | 23.44–23.44 | 10240.0 | 4096/5120 | 0.02 | 0.08247 | 1.14076 |
| final-gelu | 1 | 32.812 ± pending SD | 32.81–32.81 | 10240.0 | 5120/5120 | 0.02 | 0.10773 | 0.85525 |
| coordinate-only | 1 | 29.688 ± pending SD | 29.69–29.69 | 4096.0 | 4096/4096 | 0.01 | 0.04608 | 0.53144 |
| no-affine | 1 | 25.000 ± pending SD | 25.00–25.00 | 6144.0 | 1024/1024 | 0.01 | 0.09756 | 0.92185 |
| projection-only | 1 | 28.125 ± pending SD | 28.12–28.12 | 0.0 | 0/0 | 0.01 | 0.01558 | 0.12142 |

Identity retains theta=0 as frozen parameters and executes the same rotation arithmetic. Coordinate-only removes mixing entirely; it is the same eight-stage coordinate-affine/GELU depth. Random-fixed freezes the original random angles. Final-GELU uses one GELU after all eight affine rotation stages. No-affine removes scale/bias entirely. Projection-only removes the hidden core, retaining input projection/GELU and classifier. Frozen parameters are included in total but excluded from trainable counts.

## 6. Proper validation-selected early exit

A fixed seeded 55,000/5,000 partition of official training data is shared across all five seeds. Only validation chooses the policy: minimum mean stages subject to accuracy within 0.1 pp of the same weights' final head. The policy JSON is saved before evaluating the official test labels. Full-depth/no-gating is a fallback. This separate experiment has fewer training updates than the 60,000-example replication; it is not pooled with that table.

| Seed | Selected threshold/policy | Validation accuracy (%) | Full validation (%) | Test accuracy (%) | Full test (%) | Mean/median test stages | Core / whole ops | Batch256 / batch1 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.1 | 15.62 | 15.62 | 25.00 | 20.31 | 1.000/1.0 | 1280.0/407808.0 | 0.01369/0.13869 |

Exit counts for each frozen policy (stages 1..8):

- Seed 42: 64, 0, 0, 0, 0, 0, 0, 0

## 7. Stage rescue/damage analysis

Counts below are means across seeds on the same 10,000 official test images; these are not independent new datasets.

| Transition | Rescued mean ± SD | Damaged mean ± SD | Incorrect→incorrect mean | Correct→correct mean |
|---|---:|---:|---:|---:|
| 1→2 | 0.0 ± pending SD | 1.0 ± pending SD | 44.0 | 19.0 |
| 2→3 | 1.0 ± pending SD | 1.0 ± pending SD | 44.0 | 18.0 |
| 3→4 | 1.0 ± pending SD | 4.0 ± pending SD | 44.0 | 15.0 |
| 4→5 | 2.0 ± pending SD | 0.0 ± pending SD | 46.0 | 16.0 |
| 5→6 | 0.0 ± pending SD | 4.0 ± pending SD | 46.0 | 14.0 |
| 6→7 | 2.0 ± pending SD | 1.0 ± pending SD | 48.0 | 13.0 |
| 7→8 | 3.0 ± pending SD | 3.0 ± pending SD | 46.0 | 12.0 |

| Seed | Stage1 correct | Stage1 errors eventually correct at final | Errors ever corrected | Stage1 correct broken at final | Correct ever broken |
|---|---:|---:|---:|---:|---:|
| 42 | 20 | 3 | 6 | 8 | 8 |

## 8. 16x16 MNIST: no dense input projection

| Model/control | Seeds | Mean accuracy ± sample SD (%) | Min–max (%) | Mean core ops | Mean core trainable/total params | Mean train s | Mean batch=256 ms/sample | Mean batch=1 ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive-rotation | 1 | 4.688 ± pending SD | 4.69–4.69 | 10240.0 | 5120/5120 | 0.02 | 0.20709 | 2.48931 |
| dense | 1 | 26.562 ± pending SD | 26.56–26.56 | 131072.0 | 65792/65792 | 0.00 | 0.00964 | 0.09737 |
| low-rank | 1 | 23.438 ± pending SD | 23.44–23.44 | 32736.0 | 16640/16640 | 0.00 | 0.00989 | 0.18683 |
| rotation | 1 | 4.688 ± pending SD | 4.69–4.69 | 10240.0 | 5120/5120 | 0.01 | 0.07449 | 1.01061 |

| Model | Mean core ops | Mean whole-model ops |
|---|---:|---:|
| adaptive-rotation | 10240.0 | 51200.0 |
| dense | 131072.0 | 136192.0 |
| low-rank | 32736.0 | 37856.0 |
| rotation | 10240.0 | 15360.0 |

Fixed bilinear antialiased resizing; flatten directly to 256. No input projection or input GELU remains. All models share this representation. Adaptive uses the preset 0.90 policy. These results are separate from 28x28; resolution and model capacity both change.

## 9. Runtime optimization

Inference-only copies cache sin/cos and reuse adaptive exit confidences. Learned mathematics and original weights are preserved. All-stage logits on every official test image must match at atol=1e-6, rtol=1e-5; predictions/exits must match exactly. Five implementations are measured in seeded shuffled order within each timing repetition, at batch 1 and 256. Compilation is optional and not attempted; this comparison measures the portable eager cache path.

| Implementation | Seeds | Mean batch256 ms/sample ± SD | Mean batch1 ms/sample ± SD |
|---|---:|---:|---:|
| dense | 1 | 0.01711 ± pending SD | 0.13961 ± pending SD |
| original-rotation | 1 | 0.08160 ± pending SD | 1.45290 ± pending SD |
| original-adaptive | 1 | 0.19242 ± pending SD | 0.75898 ± pending SD |
| optimized-rotation | 1 | 0.07337 ± pending SD | 1.01372 ± pending SD |
| optimized-adaptive | 1 | 0.20151 ± pending SD | 0.77998 ± pending SD |

Maximum observed absolute logit error across verified test passes: 0. Exact exit and predicted-class agreement passed for completed runtime cases. Seed-42 CPU operator profiles (including calls/time for trig, GELU, indexing and matrix multiplies) are saved per batch size.

## 10. Failures and negative findings

No recorded execution failures; numerical and methodological limitations still apply.

Evidence-based answers to the seven research questions:

1. Across 1 paired MNIST seeds, RotationNet minus Dense accuracy averaged +1.562 pp, range +1.562 to +1.562 pp. This is descriptive consistency evidence, not a significance or noninferiority test.
2. See the separate Fashion-MNIST five-seed table; five seeds alone do not establish broad generalization.
3. Learned minus identity mean accuracy is -7.812 pp. Learned minus frozen-random is -1.562 pp. These controls, including coordinate-only and no-affine variants, test necessity; they do not uniquely identify a causal mechanism.
4. Per seed, mean 3.0 stage-1 errors are correct at the final stage, while mean 8.0 stage-1 correct predictions are wrong at the final stage. Error rescue is observed; 'difficulty' is not independently measured.
5. Validation-selected policies use mean 1.000/8 test stages. Mean test accuracy difference from the same weights' final head is +4.688 pp (range +4.688 to +4.688); mean whole-model multiply/add savings are 2.15%. A validation tolerance does not guarantee test accuracy retention.
6. Optimized RotationNet takes 4.29× Dense's batch=256 time on this CPU. A dense latency advantage remains.
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
- Fashion-MNIST is unexecuted when marked blocked. No conclusion beyond MNIST is supported.

## 12. Recommended Phase 3

First run this unchanged protocol on cached Fashion-MNIST and one harder dataset. Predefine an accuracy-retention margin and a meaningful latency target. Prioritize comparisons against projection-only, coordinate-depth and low-rank controls. If mixing remains useful without a dense input projection, implement a genuinely fused CPU/GPU structured kernel, verify numerical/exit equivalence, and compare both accuracy and end-to-end latency over more hardware and training budgets. If no-mixing controls match RotNet or direct-input accuracy collapses, narrow or stop the architectural claim.

## Reproduction and preservation

```powershell
.\.venv\Scripts\python.exe scripts\run_phase2.py --run-dir "C:\srdev\work\research\RotNet\results\phase2\smoke-falsification" --resume
```

Completed cases are skipped on resume. Failures retain original attempts; --retry-failed creates a new numbered attempt. Phase 1 results and source files are protected by a before/after SHA-256 manifest.

Phase 1 preservation: {'status': 'passed', 'files_checked': 74, 'changed': []}.
