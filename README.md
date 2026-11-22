# Gym Crowdedness Prediction

DS4420 project that predicts hourly gym occupancy using historical attendance, calendar patterns, temperature, and semester information.

The project compares autoregressive (AR) models and Bayesian negative binomial regression against simple weekly baselines. Data is cleaned and aggregated into hourly readings, and models are evaluated using mean absolute error (MAE) and root mean squared error (RMSE), including forecasts up to 24 hours ahead.

## What's included

- `data/gymData.csv`: gym attendance dataset.
- `src/data.py`: data cleaning, hourly aggregation, and evaluation splits.
- `src/ar.py`: weekly baselines and autoregressive models.
- `src/bayes.py`: Bayesian regression, prediction intervals, and AR comparisons.
- `src/eda/eda.ipynb`: exploratory data analysis.
- `results/`: saved forecast plots, error tables, and model summaries.
- `project_poster.pdf`: project poster.

## Setup and usage

From the project root, create a virtual environment:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux.

Install dependencies and run either model:

```bash
python -m pip install -r requirements.txt
python -m src.ar
python -m src.bayes
```

Both scripts read the included dataset and save outputs to `results/`. The Bayesian model uses MCMC sampling and may take longer to run. Existing plots and results can be viewed without rerunning the models.
