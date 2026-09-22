# Artifact: Red-Teaming Latent Memory

**Anonymous SaTML 2027 Submission**

## Quick Verification

```bash
cd artifact
python scripts/verify_results.py
```

Expected: `VERIFICATION SUMMARY: 31 passed, 0 failed`

## Directory Structure

```
artifact/
├── README.md
├── scripts/                       # Attack and evaluation scripts
│   ├── verify_results.py          # Verify all paper claims against result files
│   ├── g13_run_attack.py          # WMRP matched MemGen vs vanilla evaluation
│   ├── g14_run.py                 # Poison complexity sweep
│   ├── g15_run.py                 # Position control, non-numeric target
│   ├── g16_capped.py              # Capped config runs
│   ├── g18_latent_defense.py      # Latent-space defense
│   ├── g18_separability.py        # Per-feature AUC for latent defense
│   ├── g19_defenses.py           # Published defenses
│   ├── g19_fix.py                # 7B judge + 7B paraphraser
│   ├── g19_clean_accuracy.py     # Clean accuracy impact
│   ├── g19_smollm_n200.py        # SmolLM3-3B runs
│   └── g19_vanilla_uncapped_n200.py
├── results/                       # Raw result JSON files (22 files)
│   ├── g13_memgen_poison_1.json   # Qwen MemGen poison (n=200)
│   ├── g13_vanilla_poison_1.json  # Qwen vanilla poison (n=200)
│   ├── g13_memgen_clean.json     # Qwen MemGen clean (n=200)
│   ├── g13_vanilla_clean.json    # Qwen vanilla clean (n=200)
│   ├── g19_smollm_*.json         # SmolLM3 results (n=200 each)
│   ├── g14_*.json                # Poison complexity sweep
│   ├── g15_*.json                # Controls
│   ├── g18_latent_features.json  # Latent defense
│   ├── g19_fix_n50.json          # 7B LLM detection
│   ├── g19_clean_accuracy_n50.json
│   └── g19_defense_results.json
└── data/
    └── g13_correct_with_answers.json  # 200-item matched set
```

## Paper-to-Artifact Mapping

| Paper Table | Result File | Script |
|---|---|---|
| Table 2: WMRP Qwen | g13_memgen_poison_1.json, g13_vanilla_poison_1.json | g13_run_attack.py |
| Table 2: WMRP SmolLM3 | g19_smollm_memgen_poison_n200.json, g19_smollm_vanilla_poison_n200.json | g19_smollm_n200.py |
| Table 7: Position control | g15_position_doc2.json, g15_position_doc3.json | g15_run.py |
| Table 8: Specificity | g14_memgen_medium.json | g14_run.py |
| Table 9: Amplification sweep | g14_memgen_{terse,medium,long}.json, g14_vanilla_*.json | g14_run.py |
| Table 14: Defenses | g19_fix_n50.json, g19_clean_accuracy_n50.json, g19_defense_results.json, g18_latent_features.json | g19_*.py, g18_*.py |

## Verified Numbers (all 31 checks pass)

| Claim | Paper | Verified |
|---|---|---|
| Qwen MemGen ASR | 45.0% | 90/200 = 45.0% |
| Qwen Vanilla ASR | 12.0% | 24/200 = 12.0% |
| Amplification | 3.8x | 90/24 = 3.8x |
| McNemar | 75/9, chi2=50.30, p=4.3e-14 | matches |
| SmolLM3 MemGen ASR | 18.5% | 37/200 = 18.5% |
| SmolLM3 Vanilla ASR | 4.5% | 9/200 = 4.5% |
| LLM det 7B TPR | 100% | 100% |
| RobustRAG ASR | 0% | 0% |
| Latent AUC | ~0.53 | ~0.53 |

## ASR Definition

Tagged ASR: target must appear in `<answer>` tag (final_answer field).

## Dependencies

Python 3.10+, PyTorch, transformers, peft, datasets, scipy, requests, MemGen codebase, E5 retriever.
