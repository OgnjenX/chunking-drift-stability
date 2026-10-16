# Registered results

Generated from saved per-seed measurements. Interpret with docs/protocol.md.

| Model | Parameters | A old clean | B all clean | B old items | B old pairs | Recoding cosine |
|---|---:|---:|---:|---:|---:|---:|
| flat | 3456 | 1.000 | 1.000 | 1.000 | 1.000 | 0.009 |
| dense | 5760 | 1.000 | 1.000 | 1.000 | 1.000 | 0.010 |
| competitive | 5760 | 1.000 | 0.979 | 0.875 | 1.000 | 0.014 |

Competitive minus capacity-matched dense, step 100/noise 0.08:

| Schedule | Clean retention difference [95% CI] | Robustness difference [95% CI] | Lower/higher cosine drift | Full gates passed |
|---|---:|---:|---:|---|
| frozen | -0.018 [-0.042, +0.000] | -0.007 [-0.017, +0.000] | 0.091 / 0.082 | False |
| rehearsal | -0.005 [-0.013, +0.000] | -0.001 [-0.002, +0.001] | 0.078 / 0.115 | False |

Reuse masked-target difference: +0.011, 95% CI [-0.0325927734375, 0.057620239257812456]. Gate passed: False.

All intervals resample the eight paired seeds. They do not cover task or hyperparameter uncertainty.
Acquisition failures, per-seed values, lesion gates, and ceilings are in summary.json.
