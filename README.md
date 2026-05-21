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

### Running COALA

COALA takes a pre-trained ML model and a dataset as input. The user 
defines which features are mutable (actionable) and which are 
constrained (fixed). COALA then identifies the optimal counterfactual 
for each subject.

```python
from cf_search import COALA

# Initialize COALA with your model and feature categories
coala = COALA(model=your_model, 
              mutable_features=mutable_cols,
              constraint_features=constraint_cols)

# Run optimization
counterfactuals = coala.run(X)
```

### Reproducing Paper Results

Each visualization notebook corresponds to a dataset and analysis 
presented in the paper:

- `visualization_synthetic.ipynb` — synthetic dataset results
- `visualization_xgb_nhanes_diet.ipynb` — NHANES dietary analysis
- `visualization_xgb_nhanes_multi.ipynb` — NHANES multi-cell analysis (Supplementary)
- `visualization_xgb_fhs.ipynb` — Framingham Heart Study analysis (Supplementary)

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


