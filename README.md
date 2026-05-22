# MSiD

Repository with three parts of an MSiD course project based on analysis of football player data from `players_22.csv`.

## Structure

- `part1_exploratory_analysis`
  - exploratory data analysis
  - notebook: `src/exploratory_analysis.ipynb`
  - report: `report_part1_exploratory_analysis.pptx`
- `part2_regression_models`
  - comparison of regression models for predicting `overall`
  - baseline regressors, linear regression from scratch, and a PyTorch implementation
  - report: `report_part2_regression_models.pptx`
- `part3_extended_model_benchmark`
  - extended benchmark of models and experiment variants
  - separate model implementations in `models`
  - report: `report_part3_extended_model_benchmark.pptx`

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended Python version: `3.11` or newer.

## Run

```bash
cd part1_exploratory_analysis/src
python main.py
```

```bash
cd part2_regression_models/src
python main.py
```

```bash
cd part3_extended_model_benchmark
python main.py
```

## Dataset

Each part keeps its own local copy of `players_22.csv` so scripts can be run directly from their respective directories.
