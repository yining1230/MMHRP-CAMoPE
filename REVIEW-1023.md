# Code review and release audit

All 357 source files were inventoried and SHA256-hashed. Executable source was
inspected statically; no training or benchmark rerun was performed. Successful
syntax checks do not establish scientific validity.

## Corrected or consolidated

1. **Duplicate definitions and order-dependent execution.** AHO `Explain-all.ipynb`
   and `Sankey Figure.ipynb` repeatedly redefine prediction, routing, atom/BRICS
   attribution and attention functions. Heck `main/explain/explain.ipynb` has
   similar repeated versions. Replaced with one canonical Python implementation
   per dataset and a thin notebook launcher.
2. **Two-expert equal-flow Sankey artifact.** Absolute changes in two normalized
   Softmax weights have equal magnitude. New diagrams use the mean positive
   part of original-minus-perturbed weights, export signed changes and keep the
   last partial dynamic frame. Both cumulative and disjoint-block modes exist.
3. **Bin endpoints and artificial zero weights.** Old `digitize(x, edges)` logic
   omitted maximum-edge observations; empty groups were plotted as zero. New
   grouping includes the final endpoint, uses bin centers, exports group counts
   and uses NaN for empty bins.
4. **Population-versus-sample scope.** Setting `sample = dataset[1092]` did not
   change functions called with the full dataset into single-reaction analyses.
   Population summaries and fixed-reaction condition scans are now separate.
5. **Attention extraction changed the evaluated model.** Several `patched_forward`
   implementations return only self-attention output and discard the original
   residual/feed-forward/norm path. The later manual QKV implementation misses
   `norm_first`. New extraction observes the original input and evaluates the
   original attention module with its normalization convention, keeps forward
   unchanged and verifies unchanged predictions.
6. **Padding and token identity.** The text tensor is a continuous embedding
   matrix, not integer IDs. Counting its nonzero entries is not a token count.
   New cropping uses actual tokenized length and repeated tokens retain distinct
   position IDs. No new padding mask changes historical model behavior.
7. **Missing external helper files.** Heck's optimized heatmap wrapper and yield
   map script import missing parent/server scripts. Canonical entry points depend
   only on the supplied models/utils, with explicit checkpoint/data arguments.
8. **Best-model aliasing and wrong checkpoint slot.** Training notebooks stored
   live model references, which continued changing after the best epoch. Some
   appended a learning rate and then saved `best[-1]`, writing that scalar.
   The copy captures independent state_dict snapshots and saves model slot 8.
   Affected files are listed in `release_changes.json`; old checkpoints and scores
   are not retrospectively repaired.
9. **Language, paths and cached state.** Retained Chinese source comments/messages
   were translated; Chinese/garbled asset names and the dynamic-Sankey directory
   received English names. Notebook outputs were cleared, cache copies excluded,
   and notebook/data-process roots made portable. Original source names/historical
   metadata paths are Unicode-escaped in provenance records, not fabricated.
10. **README mismatch.** The previous top-level document described another
    Heck-only paper and absent directories; new documentation reflects this upload.

## Issues requiring author decisions or research reruns

| Priority | Source | Finding |
| --- | --- | --- |
| High | Whole upload | No trained `.pth`/`.pt` checkpoint. Add actual state_dicts and provenance before numerical reproduction. |
| High | Heck main training notebook and several legacy notebooks | Missing explicit `model.train()`/`model.eval()` transitions can leave dropout and BatchNorm active during evaluation. Rerun under a defined protocol. |
| High | Multiple training notebooks | `data[:split]` with `data[split+1:]` omits one reaction. Confirm split coverage/manifests; historical split behavior was not silently changed. |
| High | Training epoch selection | Selection by test R2 compromises an independent final test estimate. Use validation-based selection and an untouched final test, which requires retraining. |
| High | Data / `utils/molecule.py::smi_to_graph` | Some supplied molecular strings fail sanitization under the tested RDKit runtime. Graph conversion assumed a valid molecule. Consult `data_validation_report.json`, confirm original RDKit version and verify structures against their sources. No rows were silently repaired or dropped. |
| Medium | AHO training-performance figures | Some multiply ddG by 100 and use yield labels/0-100 reference lines. ddG is not fractional yield. New explanation exports retain raw ddG; legacy plot cells require review. |
| Medium | `utils/rxn.py::RxnSmi_to_tensor` | Unknown tokens use unrecorded random vectors. Canonical analysis records a deterministic per-sample seed; historical figures may differ without original embeddings. |
| Medium | `models/DeepLearnModel.py::BiGRU` | Hidden state follows stored `self.device`; `model.to(...)` need not update it. Canonical analysis synchronizes stored device attributes. |
| Medium | Low-error example CSVs | Lowest-error examples do not prove generalization. Confirm manuscript examples' train/test membership. New population routing does not rank by error. |
| Medium | Heck yield maps | Unconstrained regressors may predict outside 0-100%. Preserve/report this rather than claiming a physically attainable optimum. |
| Medium | Condition scans | Marginal min/max scans may cover unsupported joint combinations. Check joint training support and experimental plausibility. |
| Medium | `pkgs.txt` | Historical environment dump is platform/hardware specific and not a portable lockfile. The new dependency list is deliberately not represented as a fully tested universal pin set. |
| Release | Citation/license/data | Authors must confirm final manuscript citation, license and redistribution permissions. These cannot be inferred from code. |

## Retained and omitted material

The manifest records all 357 source files and their hashes. Only superseded
explanation source and caches were omitted from this independent copy. Scientific
datasets and reference figures/HTML/grids/tables were retained. Historical images
may reflect old implementations; retaining them does not certify reproduction.

Exploratory dataset-wide fragment summaries, token-network styling variants and
Pd-specific printouts remain in the original notebooks. The canonical analysis
retains single-reaction atom/functional-group/BRICS/text analysis and population
prediction/routing/Sankey analysis, but does not claim to reproduce every obsolete
exploratory variant. Helper-directory READMEs are superseded by the root guide.

## Verification scope

Read `validation_report.json` for syntax/language/source-hash checks,
`data_validation_report.json` for molecular parsing, and `smoke_test_report.json`
for actual integration checks with temporary synthetic weights and valid small
fixtures drawn from the supplied data. Synthetic weights are not uploaded as
trained models and prove no predictive accuracy. Full pretrained inference,
historical figures and benchmark metrics remain unverified without checkpoints.
