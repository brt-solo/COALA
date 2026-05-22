# COALA: Counterfactual Optimization for Actionable interpretabiLity in AI

COALA is a framework for identifying optimal counterfactuals across a 
population of subjects from a pre-trained machine learning model, 
revealing which constrained features determine what the optimal 
counterfactual is for a given individual. COALA is designed to support 
personalized intervention strategies in biomedical research.

## Repository Structure
COALA/

├── cf_search/          # Core COALA implementation

├── models/             # Trained models (synthetic and real datasets)

├── public_datasets/    # NHANES and Framingham datasets

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

Before running, edit the config section at the top of `run_coala.py` 
to point to your files:

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
- `MAX_INDIVIDUALS` — maximum number of subjects to process (default: 200)
- `METHOD` — crossover method; options are `uniform`, `single_point`, 
  `simulated_binary`, or `random_mutation` (default: `uniform`)

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
python run_coala.py
```

Output is saved as `counterfactuals_multi.pkl` in the specified output 
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

Synthetic data and trained models are included in this repository.

## Citation

If you use COALA in your research, please cite:

```bitex
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

