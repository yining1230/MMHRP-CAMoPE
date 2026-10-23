# MMHRP-CAMoPE: organized source release, 2026-10-23

This independent copy contains the supplied multimodal reaction-prediction
code, data, legacy experiments and reference figures. The original directory
was not modified. Read `REVIEW-1023.md` before scientific reuse.

## Consolidated explanation entry points

* `exp/AHO/explain/Explain-1023.py`: AHO; default shuffled position 1092,
  shuffle seed 1. Numerical inputs: temperature, pressure, S/C ratio.
* `exp/Heck_Benchmark/main/explain/Explain-1023.py`: HeckLit; default shuffled
  position 3116, shuffle seed 5. Numerical inputs: temperature and time.
* The `Explain-1023.ipynb` beside each script is a thin launcher. There is one
  maintained algorithm implementation per dataset, not duplicate notebook code.

No trained checkpoints were present in the supplied upload. Supply your trusted
state_dict checkpoint matching the dataset architecture. Examples from the
project root:

```bash
python exp/AHO/explain/Explain-1023.py --help
python exp/AHO/explain/Explain-1023.py --checkpoint /path/to/aho.pth --tasks sample conditions
python exp/Heck_Benchmark/main/explain/Explain-1023.py --checkpoint /path/to/heck.pth --tasks sample conditions
```

Install scientific dependencies listed in `requirements-analysis.txt` in an
environment matched to your checkpoint. The old `pkgs.txt` is an uncurated
historical environment dump, not a portable lockfile. RDKit compatibility with
the supplied molecular strings must be checked; see `data_validation_report.json`.

The default output is a new timestamped directory beside the script. A supplied
`--output-dir` must not already exist. Importing a script does not run inference.

## Tasks

| `--tasks` value | Analysis |
| --- | --- |
| `sample` | Observed value, prediction, original conditions and expert weights for the selected reaction |
| `conditions` | Single-reaction two-condition prediction and expert maps, CSV grids and shared expert color scale |
| `routing` | Population condition-bin means, counts and expert usage |
| `sankey` | Directional support, signed changes, raw arrays, static HTML and cumulative/block animation HTML |
| `atoms` | Node-feature masking for both graph branches, signed CSV scores and absolute-score molecular coloring |
| `fragments` | Functional-group and BRICS atom-set masking, tables and bar plots |
| `text` | Original-parameter attention, position-indexed token scores, heatmap and top dependency edges |
| `predictions` | Population predictions, observed labels and source reaction identifiers |
| `all` | Every task above |

Population tasks use the first `--max-samples` reactions in shuffled order;
default 1000, or 0 for the entire dataset. There is no low-error sample selection.
Single-reaction scans use the supplied dataset's marginal condition range unless
`--temperature-range MIN MAX` / `--secondary-range MIN MAX` is supplied.

AHO ddG retains its original target scale (kcal/mol). Heck fractional predictions
are multiplied by 100 for prediction figures/tables and are not clipped. The
third AHO condition is S/C, not time. Indices are zero-based after shuffling,
not the original dataset row identifiers.

The Sankey value is `mean(max(original_weight - perturbed_weight, 0))`.
Graph perturbation zeros node features in both branches and preserves topology;
text perturbation zeros the embedding sequence; numerical perturbation replaces
conditions with means across the supplied dataset. Expert IDs are model-specific.
Routing support is not a contribution to predictive accuracy.

Condition-bin means compare different reactions and do not isolate a causal
condition effect. Fixed-reaction scans describe model responses rather than
experimentally verified optima. Marginal ranges need not have joint training
support. Attention associations do not establish chemical causality.

Each run records data/checkpoint/vocabulary hashes, seeds, reaction identifiers,
population size, bins, ranges and status. Unknown-token vectors are deterministic
per sample and independent of task order. Historical unrecorded random vectors
cannot be guaranteed identical; exact reproduction needs original input
embeddings as well as the trained checkpoint.

## Package and verification

`data/` contains unchanged scientific data. `data_process/`, `models/`, `utils/`
and training/baseline/ablation notebooks are retained. Training notebooks have
English source, portable root discovery, cleared outputs and corrected
best-checkpoint storage. Historical split/evaluation/model-selection protocols
are flagged in the review rather than silently redefined.

* `release_manifest.json`: original hashes, English renames, omitted caches and consolidated scripts.
* `release_changes.json`: per-file checkpoint-storage corrections.
* `validation_report.json`: code syntax, literal-Chinese and source-integrity checks.
* `smoke_test_report.json`: functional checks with synthetic weights; not accuracy evidence.
* `data_validation_report.json`: molecular parsing checks under the tested RDKit runtime.

Historical images/HTML/arrays/tables are reference artifacts, not verification of
the revised algorithms. Old helper READMEs are superseded by this README; their
scripts were consolidated. Exploratory notebook variants remain recoverable in
the untouched original directory.

The former README described an unrelated prior Heck-only release and has been
replaced. Authors must provide the final manuscript citation, trained weights,
license and permission to redistribute third-party data/code before publication.
No scientific metrics, publication identifier or license has been invented.
