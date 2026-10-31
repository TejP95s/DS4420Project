# Gym Crowdedness Prediction Models

## Project Structure
project/
│
├── src/                        # Source code and reusable modules
│   ├── ar.py                   # Manual implementation of the AR model(s)
│   ├── bayes.py                # Bayes model serving as the 2nd model for the problem
│   └── data.py                 # loads, cleans and prepares the hourly dataset
│
├── data/                       # Datasets used in the project
│   └── gymData.csv             # main dataset with 62.5k rows
│
├── notebooks/                  # Jupyter notebooks for exploration
│
├── results/                    # Outputs and project findings(only going to mention the plots here)
│   └── ar_forecast_week.png    # AR forecast vs actual chart
|   ├── bayes_effect.png        # Posterior Bayes effect percents & hour of day prediction plot 
|   ├── bayes_forecast_week.png # Forecast for week of 2016-10-11
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Files and dirs not on git
└── README.md                   # Project overview and documentation

## Setup

### Mac/Linux
- python3 -m venv .venv
- source .venv/bin/activate

### Windows
- python -m venv .venv
- .venv\Scripts\Activate.ps1
If PowerShell blocks the activation script, run ```Set-ExecutionPolicy -Scope CurrentUser RemoteSigned``` once and try again. You’ll know it worked when (.venv) appears at the start of your prompt.

Upgrade pip: ```python -m pip install --upgrade pip```

Install requirements: ```pip install -r requirements.txt```