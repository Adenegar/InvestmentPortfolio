# Portfolio Diversification Strategies

This project explores portfolio diversification strategies and tests using historical data and Monte Carlo simulation. We explored using financial data as a dimension for diversification and compared it to random and industry stratified portfolios. 

## Usage

### Data Exploration and Visualization
- ratios.ipynb

### Run Simulations
- portfolios.ipynb
- run_simulation.py

prerequisite: run the first two cells in portfolios.ipynb to setup df_results.csv if not already setup. 

Basic run configuration:
```
cd src/
python run_simulation.py Random
```

Additional configuration:
```
python run_simulation.py Random --num_simulations 50 --simulation_type bootstrap --simulation_length 3y --num_stocks 10
```

More details can be found in the run_simulation.py file.

- run_batch_sims.sh (Mac/Linux batch calls to run_simulation.py)

### Evaluate Results
- evaluate.ipynb

### Helper Files (used by other files, not intended to be directly used)
- setup.py
- tools.py
- ratios.py