# MMHRP-CAMoPE

MMHRP-CAMoPE is a multimodal framework for chemical reaction outcome prediction. It integrates molecular graphs, SMILES representations, and numerical reaction conditions through condition-aware routing and prompt-level experts.

This repository provides code for data preprocessing, model training and evaluation, ablation studies, reaction-level interpretability analysis, and expert routing analysis on the AHO-Clean and HeckLit-Clean datasets.

## Description

The framework combines three sources of reaction information:

- **Graph modality:** molecular graphs describing the reaction components.
- **Text modality:** tokenized reaction SMILES represented by embedding sequences.
- **Numerical modality:** continuous experimental conditions, including temperature, pressure, substrate-to-catalyst ratio, or reaction time, depending on the dataset.

CAMoPE uses these representations to calculate expert routing weights and construct condition-aware prompts for reaction outcome prediction. The repository includes experiments for predicting enantioselectivity-related energy differences in AHO reactions and reaction yields in Heck reactions.

## Main Packages

The main packages used by the analysis scripts and notebooks are:

- PyTorch
- PyTorch Geometric
- RDKit
- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- Plotly
- openpyxl
- tqdm
- lxml
- Beautiful Soup

The direct dependencies for the consolidated analysis scripts are listed in [`requirements-analysis.txt`](requirements-analysis.txt). From the repository root, install them with:

```bash
python -m pip install -r requirements-analysis.txt
```

Use PyTorch and PyTorch Geometric builds compatible with your CPU or CUDA environment. Molecular parsing also depends on the RDKit version; check compatibility with the supplied data before running the full analysis.

The original environment package list is retained in [`pkgs.txt`](pkgs.txt). Some historical fingerprint-generation and baseline experiments require additional packages, such as RXNFP, DRFP, or other machine-learning libraries.

## Data

The datasets and associated files are located in the [`data`](data) directory.

| Dataset | Data file | Prediction target | Numerical conditions |
| --- | --- | --- | --- |
| AHO-Clean | `data/AHO/2AHO_all_condition.xlsx` | Enantioselectivity-related energy difference, ΔΔG‡ (kcal/mol) | Temperature, pressure, and substrate-to-catalyst ratio (S/C) |
| HeckLit-Clean | `data/Heck/Heck_data_clean.xlsx` | Reaction yield | Temperature and reaction time |

Additional files contain reaction fingerprints and data associated with the historical benchmark experiments. Data-processing utilities are provided in [`data_process`](data_process).

For AHO, the model uses the supplied `ddG` target directly. For HeckLit, model training uses fractional yields, while the consolidated analysis scripts convert prediction outputs to percentages for presentation.

## Models

Model architectures are defined in [`models/DeepLearnModel.py`](models/DeepLearnModel.py), including the multimodal model, baseline architectures, and modality-ablation variants. The CAMoPE model implementation used by the consolidated explanation scripts is `MMHRP_GCL_tri_MoPE_pdall`.

Other model-related files include:

- `models/fds.py`: feature distribution smoothing utilities.
- `models/smi2vec.py`: SMILES embedding utilities.
- `models/AHO_vocab.txt`: the AHO SMILES embedding vocabulary.
- `models/HeckLit_vocab.txt`: the HeckLit SMILES embedding vocabulary.

**Trained model checkpoints are not included in the current package.** Running prediction or explanation requires a compatible trained PyTorch `state_dict` checkpoint. AHO and HeckLit use different model configurations, so each analysis must use its corresponding checkpoint.

## Notebooks

The main experiments are organized under [`exp`](exp).

| Directory or notebook | Purpose |
| --- | --- |
| `exp/AHO/AHO_MMHRP-GCL-tri-mope.ipynb` | AHO multimodal model training and evaluation |
| `exp/Heck_Benchmark/main/Heck_MMHRP-GCL-tri-MOpE-pdall-gpu-IR_64.ipynb` | HeckLit multimodal model training and evaluation |
| `exp/AHO/baseline/` | AHO baseline experiments |
| `exp/Heck_Benchmark/baseline/` | HeckLit baseline experiments |
| `exp/AHO/ablation/` | AHO modality-ablation experiments |
| `exp/Heck_Benchmark/ablation/` | HeckLit modality-ablation experiments |
| `exp/AHO/explain/Explain-1023.ipynb` | Notebook launcher for AHO explanation and routing analysis |
| `exp/Heck_Benchmark/main/explain/Explain-1023.ipynb` | Notebook launcher for HeckLit explanation and routing analysis |

Open the notebooks in Jupyter with the working directory inside the repository. The analysis notebooks call the corresponding `Explain-1023.py` scripts, which contain the maintained implementations.

Additional historical experiments are retained in `exp/Heck_JCP/` and `exp/Heck_Other/`.

## Interpretability and Expert Routing Analysis

The consolidated analysis scripts provide reaction-level explanations and model-level routing analyses.

| Task | Description |
| --- | --- |
| `sample` | Report the observed value, prediction, original conditions, and expert routing weights for a selected reaction |
| `conditions` | Generate prediction and expert routing response maps by scanning two numerical conditions for a selected reaction |
| `routing` | Calculate average expert routing weights across condition ranges and summarize expert usage |
| `sankey` | Generate modality-to-expert directional routing support diagrams and dynamic visualizations |
| `atoms` | Evaluate prediction changes after masking individual graph-node features |
| `fragments` | Evaluate prediction changes after masking functional-group or BRICS fragment atom sets |
| `text` | Extract and visualize attention associations between SMILES token positions |
| `predictions` | Export predictions and observed values for the selected population of reactions |
| `all` | Run all tasks listed above |

### AHO-Clean

From the repository root:

```bash
python exp/AHO/explain/Explain-1023.py --help
```

To analyze a selected reaction, replace the checkpoint placeholder with an existing trained checkpoint:

```bash
python exp/AHO/explain/Explain-1023.py --checkpoint /path/to/aho_state_dict.pth --sample-index 1092 --tasks sample conditions
```

### HeckLit-Clean

```bash
python exp/Heck_Benchmark/main/explain/Explain-1023.py --help
```

```bash
python exp/Heck_Benchmark/main/explain/Explain-1023.py --checkpoint /path/to/heck_state_dict.pth --sample-index 3116 --tasks sample conditions
```

### Population-Level Routing Analysis

For example, to calculate condition-bin routing statistics and generate Sankey visualizations for the first 1,000 shuffled AHO reactions:

```bash
python exp/AHO/explain/Explain-1023.py --checkpoint /path/to/aho_state_dict.pth --tasks routing sankey --max-samples 1000
```

Set `--max-samples 0` to analyze the full dataset. Sample indices refer to zero-based positions **after shuffling**, rather than the original reaction identifiers. The default shuffle seeds are 1 for AHO and 5 for HeckLit.

Results are saved to a new timestamped directory beside the script. A custom `--output-dir` can also be supplied, provided that the directory does not already exist. Outputs include figures, numerical tables, and a metadata record containing the analysis settings and input-file hashes.

## Repository Structure

```text
MMHRP-CAMoPE/
├── data/                       # Reaction datasets and fingerprint files
├── data_process/               # Data preparation and analysis utilities
├── exp/
│   ├── AHO/                    # AHO training, baselines, ablations, and explanations
│   ├── Heck_Benchmark/         # HeckLit training, baselines, ablations, and explanations
│   ├── Heck_JCP/               # Additional historical benchmark experiments
│   └── Heck_Other/             # Additional historical experiments
├── models/                     # Model architectures and SMILES vocabularies
├── utils/                      # Molecular and reaction preprocessing functions
├── requirements-analysis.txt   # Direct analysis dependencies
├── pkgs.txt                    # Historical environment package list
└── README.md
```

## General Remarks

Condition-bin plots summarize routing weights across different reactions. In contrast, single-reaction response maps vary specified conditions while retaining the selected reaction's other inputs. These analyses address different questions and should be interpreted accordingly.

Sankey links represent average positive routing-weight changes relative to modality-specific reference inputs. They characterize routing sensitivity rather than direct contributions to predictive accuracy. Similarly, attention visualizations describe model associations and do not establish chemical causality.

Reproducing a numerical result requires matching the dataset, preprocessing, model configuration, trained checkpoint, and random settings. The detailed code review, molecular parsing checks, and functional-test scope are documented in [`REVIEW-1023.md`](REVIEW.md), [`data_validation_report.json`](data_validation_report.json), and [`smoke_test_report.json`](smoke_test_report.json). Historical figures are retained as reference outputs; their presence alone does not verify reproduction by the current scripts.
