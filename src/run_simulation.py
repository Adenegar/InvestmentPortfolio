"""
CLI entry point for building portfolios and running simulations.

Prerequisite: df_results.csv stored in the same directory as this file, containing composite financial metrics. Currently, you can do this by running the first two cells in portfolios.ipynb.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import okama as ok
import time
import numpy as np
from tools import save_portfolio_results

# Load your data
df_results = pd.read_csv("df_results.csv")

# add: helper to parse length strings
def _parse_length_to_years(length: str) -> float:
    if length.endswith("m"):
        return int(length[:-1]) / 12
    # assume 'y'
    return float(length[:-1])

def ticker_policy(df_results, policy="Industry", sample_size=15, random_state=42):
    match policy:
        case "Random":
            return df_results.sample(sample_size, random_state=random_state)["ticker"]
        case "Industry":
            industry_counts = df_results['Industry'].value_counts()
            selected_industries = industry_counts.sample(min(11, sample_size), random_state=random_state).index.tolist()
            industry_grouped_portfolio = df_results[df_results['Industry'].isin(selected_industries)].groupby('Industry').apply(
                lambda x: x.sample(1, random_state=random_state)
            ).reset_index(drop=True)
            additional_industries = industry_counts[~industry_counts.index.isin(selected_industries)].sample(max(0, sample_size - 11), random_state=random_state).index.tolist()
            additional_stocks = df_results[df_results['Industry'].isin(additional_industries)].groupby('Industry').apply(
                lambda x: x.sample(1, random_state=random_state)
            ).reset_index(drop=True)
            industry_grouped_portfolio = pd.concat([industry_grouped_portfolio, additional_stocks]).reset_index(drop=True)
            return industry_grouped_portfolio["ticker"]
        case "Base":
            base_portfolio = df_results.groupby('cluster').apply(
                lambda x: x.sample(min(len(x), 3), random_state=random_state),
                include_groups=False
            ).reset_index(drop=True)
            if len(base_portfolio) < sample_size:
                additional_stocks = df_results[~df_results['ticker'].isin(base_portfolio['ticker'])].sample(
                    sample_size - len(base_portfolio), random_state=random_state
                )
                base_portfolio = pd.concat([base_portfolio, additional_stocks]).reset_index(drop=True)
            return base_portfolio["ticker"]
        case "Base-High":
            base_high = df_results.groupby('cluster').apply(
                lambda x: x.nlargest(3, 'risk'),
                include_groups=False
            ).reset_index(drop=True)
            if len(base_high) < sample_size:
                additional_stocks = df_results[~df_results['ticker'].isin(base_high['ticker'])].nlargest(
                    sample_size - len(base_high), 'risk'
                )
                base_high = pd.concat([base_high, additional_stocks]).reset_index(drop=True)
            return base_high["ticker"]
        case "Base-Low":
            base_low = df_results.groupby('cluster').apply(
                lambda x: x.nsmallest(3, 'risk'),
                include_groups=False
            ).reset_index(drop=True)
            if len(base_low) < sample_size:
                additional_stocks = df_results[~df_results['ticker'].isin(base_low['ticker'])].nsmallest(
                    sample_size - len(base_low), 'risk'
                )
                base_low = pd.concat([base_low, additional_stocks]).reset_index(drop=True)
            return base_low["ticker"]
        case "LowRisk":
            low_risk_portfolio = df_results[df_results["risk"] < df_results["risk"].quantile(0.33)].sample(sample_size, random_state=random_state)
            return low_risk_portfolio["ticker"]
        case "MediumRisk":
            med_df = df_results[
                (df_results["risk"] >= df_results["risk"].quantile(0.34)) & 
                (df_results["risk"] < df_results["risk"].quantile(0.66))
            ]
            med_risk_portfolio = med_df.sample(min(sample_size, len(med_df)), random_state=random_state)
            return med_risk_portfolio["ticker"]
        case "HighRisk":
            high_risk_portfolio = df_results[df_results["risk"] >= df_results["risk"].quantile(0.67)].sample(sample_size, random_state=random_state)
            return high_risk_portfolio["ticker"]
        case _:
            raise ValueError(f"Unknown policy: {policy}")

def extract_return(wealth_index, start_date, end_date):
    start_val = wealth_index.loc[start_date, wealth_index.columns[0]]
    end_val = wealth_index.loc[end_date, wealth_index.columns[0]]
    overall = end_val / start_val - 1

    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    years = (ed - sd).days / 365.25
    annual = (end_val / start_val) ** (1 / years) - 1

    yoy = {}
    for year in range(int(start_date[:4]), int(end_date[:4])):
        d1, d2 = f"{year}-12", f"{year+1}-12"
        try:
            ret = wealth_index.loc[d2, wealth_index.columns[0]] / wealth_index.loc[d1, wealth_index.columns[0]] - 1
        except Exception:
            ret = None
        yoy[f"{year}-{year+1}"] = ret

    return overall, annual, yoy

# update signature: require sample_size
def build_portfolio(policy_type, sample_size, random_state=42):
    tickers = ticker_policy(df_results, policy_type, sample_size, random_state)
    assets = [f"{ticker}.US" for ticker in tickers]
    weights = [1/len(assets)] * len(assets)
    return ok.Portfolio(
        assets=assets,
        weights=weights,
        ccy='USD',
        rebalancing_period='month',
        first_date='2003-09',
        last_date='2024-02'
    )

# update to accept sample_size & length
def simulate_return(i, policy_type, sample_size, sim_length):
    years = _parse_length_to_years(sim_length)
    pf = build_portfolio(policy_type, sample_size, random_state=i)
    # if less than 1 year, simulate full 1‑year path and pick month
    if years < 1:
        months = int(years * 12)
        wealth = pf.monte_carlo_wealth(distr='norm', years=1, n=1)
        start_val = wealth.iloc[0, 0]
        end_val   = wealth.iloc[months, 0]
    else:
        wealth = pf.monte_carlo_wealth(distr='norm', years=int(years), n=1)
        start_val = wealth.iloc[0, 0]
        end_val   = wealth.iloc[-1, 0]
    overall = end_val / start_val - 1
    ann     = (end_val / start_val) ** (1 / years) - 1
    growth  = end_val / start_val
    return {
        "start_value": 10000,
        "end_value":   10000 * growth,
        "overall_ret": overall,
        "ann_ret":     ann,
        "yoy_returns": {}
    }

# update to accept sample_size & length
def simulate_bootstrap(i, policy_type, sample_size, sim_length):
    pf = build_portfolio(policy_type, sample_size, random_state=i)
    wealth = pf.wealth_index
    # total months from simulation_length
    total_months = int(_parse_length_to_years(sim_length) * 12)
    idx = wealth.index.to_timestamp()
    max_start = len(idx) - total_months - 1
    start_pos = np.random.randint(0, max_start + 1)
    start_date = idx[start_pos].strftime("%Y-%m")
    end_date   = idx[start_pos + total_months].strftime("%Y-%m")
    overall, ann, yoy = extract_return(wealth, start_date, end_date)
    start_val = wealth.loc[start_date, wealth.columns[0]]
    end_val   = wealth.loc[end_date,   wealth.columns[0]]
    growth    = end_val / start_val
    return {
        "start_value": 10000,
        "end_value":   10000 * growth,
        "overall_ret": overall,
        "ann_ret":     ann,
        "yoy_returns": yoy
    }

def main(policy_type, num_simulations, simulation_type, simulation_length, num_stocks):
    start_time = time.perf_counter()

    with ThreadPoolExecutor() as executor:
        if simulation_type == "monte_carlo":
            results = list(
                executor.map(
                    lambda i: simulate_return(i, policy_type, num_stocks, simulation_length),
                    range(num_simulations),
                )
            )
            results_series = pd.Series(results)
            print(results_series)
        elif simulation_type == "bootstrap":
            results = list(
                executor.map(
                    lambda i: simulate_bootstrap(i, policy_type, num_stocks, simulation_length),
                    range(num_simulations),
                )
            )
            print(pd.DataFrame(results).head())  # Just print a preview
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")

    save_portfolio_results(
        portfolio_name=f"{policy_type.lower()}_{simulation_length}_{num_stocks}s_portfolio",
        results=results,
        simulation=simulation_type
    )

    elapsed = time.perf_counter() - start_time
    print(f"With threads executed in {elapsed:0.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulations with a specified policy type.")
    parser.add_argument("policy_type", type=str, help="The policy type to use (e.g., 'Random').")
    parser.add_argument("--num_simulations", type=int, default=5, help="Number of simulations to run.")
    parser.add_argument("--simulation_type", type=str, choices=["monte_carlo", "bootstrap"], default="monte_carlo", help="Simulation type to use.")
    parser.add_argument("--simulation_length", type=str, choices=["3m", "6m", "1y", "3y", "5y", "10y"], default="5y", help="Length of simulation")
    parser.add_argument("--num_stocks", type=int, default=15)

    args = parser.parse_args()
    main(
        args.policy_type,
        args.num_simulations,
        args.simulation_type,
        args.simulation_length,
        args.num_stocks,
    )