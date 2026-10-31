# scDHF

This repository contains the implementation of **scDHF**, a dual hypergraph framework for cell-type classification in single-cell RNA sequencing data.

## 1. Introduction

Single-cell RNA sequencing (scRNA-seq) enables the study of gene expression at
single-cell resolution. Cell-type classification requires representations that
capture relationships among cells and among genes.

scDHF constructs two complementary hypergraphs: a cell hypergraph based on
expression similarity and a gene hypergraph based on reduced gene-expression
profiles. Hypergraph convolution learns representations in each branch, and
expression-weighted projection maps gene representations back to the cell space.
The resulting cell representations are concatenated for classification. A
consistency loss encourages agreement between the two branches during training.

The implementation evaluates scDHF using stratified cross-validation. Within
each fold, standardization is fitted on training cells, and the best model is
selected using validation Macro-F1. Hypergraph construction is transductive:
features from all cells are available when constructing the hypergraphs, while
supervised classification loss uses training labels only.

## 2. About this repository

The source code is organized by function:

```text
.
├── README.md
├── requirements.txt
├── main.py
├── data/
├── results/
└── src/
    ├── config.py
    ├── pipeline.py
    ├── data/
    │   └── preprocessing.py
    ├── graphs/
    │   └── hypergraph.py
    ├── models/
    │   ├── layers.py
    │   └── scdhf.py
    ├── training/
    │   └── cross_validation.py
    ├── evaluation/
    │   └── metrics.py
    └── utils/
        └── reproducibility.py
```

`main.py` is the entry point. Model definitions, hypergraph construction,
preprocessing, training, and evaluation are implemented in the corresponding
subdirectories of `src/`.

## 3. Usage

### 3.1. Installation

After cloning this repository, enter its root directory and install the required
packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The implementation uses NumPy, pandas, PyTorch, scikit-learn, and openpyxl. CUDA
is used automatically when available; otherwise, the model runs on CPU.

### 3.2. Datasets

Place each input dataset in `data/` as a CSV file. Rows represent cells, the
first column contains cell-type labels, and the remaining columns contain
gene-expression values. The CSV must include a header row. For example:

```csv
label,GeneA,GeneB,GeneC
type_1,0,3,1
type_2,5,0,2
```

Datasets are supplied separately by the user. Preprocessing removes genes with
zero expression in every cell, applies `log1p` when the data are nonnegative and
the maximum expression exceeds 20, and selects genes by dispersion when the
number of genes exceeds `n_hvg`.

### 3.3. Example

For a dataset stored at `data/Adam.csv`, run:

```bash
python3 main.py --dataset Adam
```

For another dataset, replace `Adam` with its filename stem:

```bash
python3 main.py --dataset my_dataset
```

Each invocation processes one dataset. Inputs are read from `data/`, and outputs
are saved directly to `results/`. Custom directories can also be specified:

```bash
python3 main.py --dataset my_dataset --data_dir /path/to/data --results_dir /path/to/results
```

### 3.4. Parameters

The following command-line options control the experiment:

| Option | Description | Default |
| --- | --- | --- |
| `--dataset` | Dataset filename, with or without `.csv` | Required |
| `--n_hvg` | Maximum number of genes retained by dispersion selection | `2048` |
| `--n_neighbors` | Number of neighbors for cell hypergraphs | `20` |
| `--gene_neighbors` | Number of neighbors for gene hypergraphs | `20` |
| `--gene_svd_dim` | Requested dimension of reduced gene features | `64` |
| `--hidden_dim` | Hidden cell-representation dimension | `128` |
| `--gene_dim` | Gene-embedding dimension | `64` |
| `--dropout` | Dropout probability | `0.2` |
| `--lambda_cons` | Consistency-loss weight | `0.1` |
| `--epochs` | Maximum number of training epochs | `200` |
| `--lr` | Learning rate | `0.001` |
| `--weight_decay` | AdamW weight decay | `0.001` |
| `--patience` | Early-stopping patience | `10` |
| `--n_splits` | Number of stratified cross-validation folds | `5` |
| `--val_size` | Validation fraction of the outer training partition | `0.2` |
| `--seed` | Base random seed | `42` |

Each fold uses `seed + fold`, with fold numbers starting at 1. The original
numerical defaults and seed logic are preserved.

All available options can be displayed with:

```bash
python3 main.py --help
```

### 3.5. Outputs

For each dataset, the following files are generated in `results/`:

- `<dataset>_scDHF_5fold_results.csv`: validation and test metrics for each fold.
- `<dataset>_scDHF_5fold_predictions.xlsx`: test predictions and class probabilities,
  with one worksheet per fold.
- `<dataset>_scDHF_5fold_summary.csv`: dataset information and the mean and standard
  deviation of test metrics across folds.

## References and contact

Please cite the accompanying scDHF paper when using this implementation in your
research. For questions about the code, open an issue in this repository.
