# COALA: Counterfactual Optimization for Actionable interpretabiLity in AI

COALA is a framework for identifying optimal counterfactuals across a 
population of subjects from a pre-trained machine learning model, 
revealing which constrained features determine what the optimal 
counterfactual is for a given individual. COALA is designed to support 
personalized intervention strategies in biomedical research.

## Repository Structure
COALA/

├── cf_search/          # Core COALA implementation

├── models/             # Synthetic trained model; real-dataset train/test splits only (see Data Availability)

├── public_datasets/    # Scripts and feature-category configs for the NHANES/Framingham analyses (raw data not included — see Data Availability)

├── synthetic/          # Synthetic dataset and data generation scripts

├── visualization_synthetic.ipynb         # Figures for synthetic dataset analysis

├── visualization_xgb_nhanes_diet.ipynb   # Figures for NHANES dietary analysis

├── visualization_xgb_nhanes_multi.ipynb  # Figures for NHANES multi-cell analysis

├── visualization_xgb_fhs.ipynb           # Figures for Framingham analysis

└── requirements.txt    # Python dependencies


## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/brt-solo/COALA.git
cd COALA
pip install -r requirements.txt
```

## Usage

### Prerequisites

Ensure you have a trained ML model saved as a `.pkl` file and your 
dataset split into train and test CSV files.

### Configuration

There is no single `run_coala.py` entry point — instead, each dataset
has its own runner script in `public_datasets/`:

- `public_datasets/run_coala_xgb_fhs.py` — Framingham Heart Study
- `public_datasets/run_coala_xgb_nhanes_diet.py` — NHANES dietary analysis
- `public_datasets/run_coala_xgb_nhanes_multi.py` — NHANES multi-cell analysis

Before running, edit the config section at the top of the relevant
script to point to your files:

```python
MODEL_PATH      = "path/to/your/model.pkl"
TRAIN_PATH      = "path/to/X_train.csv"
REFERENCE_PATH  = "path/to/X_test.csv"
FEATURE_JSON    = "path/to/feature_categories.json"
OUTPUT_DIR      = "path/to/output/directory"
```

Key parameters:
- `INIT_POP` — number of random counterfactuals generated in the 
  initialization phase (default: 1000)
- `MAX_ITER` — maximum number of iterations per subject (default: 10000)
- `MAX_INDIVIDUALS` — maximum number of subjects to process (varies by 
  script: 500 for the FHS and NHANES dietary runners, 200 for the 
  NHANES multi-cell runner)
- `METHOD` — crossover method; options are `uniform`, `single_point`, 
  or `sbx` (simulated binary crossover) (default: `uniform`)
- `MUTATION_RATE` — probability of applying random mutation on top of 
  crossover; set to `None` or `0` to disable it (default: `None`)

### Feature Categories

Feature categories are defined in a JSON file that specifies which 
features belong to which group. For example:

```json
{
  "dietary": ["Energy (kcal)", "Protein (g)", "Total fat (g)"],
  "clinical": ["Age (years)", "Waist circumference (cm)"]
}
```

Features in mutable categories will be optimized; all others are held 
constant as constraint features.

### Running COALA

```bash
python public_datasets/run_coala_xgb_fhs.py
```

Output is saved as `counterfactuals.pkl` in the specified output 
directory — a dictionary mapping each subject index to a DataFrame of 
their optimal counterfactuals across cells.

### Reproducing Paper Results

To reproduce the figures in the paper, run the corresponding notebook 
after generating counterfactuals:

- `visualization_synthetic.ipynb` — synthetic dataset (Figures 2, S1)
- `visualization_xgb_nhanes_diet.ipynb` — NHANES analysis (Figures 3–6)
- `visualization_xgb_nhanes_multi.ipynb` — multi-cell analysis (Supplementary)
- `visualization_xgb_fhs.ipynb` — Framingham analysis (Supplementary)

## Data Availability

The NHANES 2017–2018 dataset is publicly available from the 
[CDC/NCHS](https://www.cdc.gov/nchs/nhanes/index.htm).

The Framingham Heart Study dataset was obtained from the publicly 
available MIT OpenCourseWare repository.

Synthetic data and the synthetic trained model are included in this 
repository. Raw NHANES/FHS data and the real-dataset trained models 
are not included — obtain the data from the sources above, then use 
`public_datasets/xgb_fhs.py` / `xgb_nhanes_diet.py` to train and save 
the corresponding models before running COALA on them.

## Citation

If you use COALA in your research, please cite:

```bibtex
@article{han_2025,
  author  = {Han, Bryant and Duan, Qingling and Hu, Ting},
  title   = {Identifying intervention strategies from machine learning 
             models with {COALA}: a counterfactual optimization framework},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.07.18.664723},
  note    = {Preprint}
}
```
## License

This project is licensed under the MIT License.

