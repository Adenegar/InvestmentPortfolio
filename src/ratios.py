"""
Compute financial ratios using Yahoo Finance and Okama.
"""
import time
import logging
import pandas as pd
import yfinance as yf
import okama as ok
from concurrent.futures import ThreadPoolExecutor, as_completed

# Policy variable: change this to use a different baseline year.
DEFAULT_YEAR = "2023"
PRIOR_YEAR = str(int(DEFAULT_YEAR) - 1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_financial_data(ticker_symbol):
    """
    Fetch balance sheet, income statement, and cash flow data for a given ticker.
    
    Parameters:
        ticker_symbol (str): The ticker symbol.
        
    Returns:
        tuple: (income_stmt, balance_sheet, cash_flow, tick, stock)
    """
    try:
        tick = yf.Ticker(ticker_symbol)
        balance_sheet = tick.balance_sheet
        income_stmt = tick.income_stmt
        cash_flow = tick.cashflow
        
        if balance_sheet.empty or income_stmt.empty or cash_flow.empty:
            raise ValueError(f"Financial statements not available for {ticker_symbol}")

        # Optional: Rename columns if necessary
        income_stmt.rename(columns={"" : "Account"}, inplace=True)
        
        stock = ok.Asset(ticker_symbol + ".US")
    except Exception as e:
        logger.error(f"Error fetching financial data for {ticker_symbol}: {e}")
        raise
    return income_stmt, balance_sheet, cash_flow, tick, stock

def extract_value(df, row, col):
    """
    Helper function to extract a value from a DataFrame.
    
    If the value cannot be found, pd.NA is returned.
    
    Parameters:
        df (DataFrame): The DataFrame to extract from.
        row (str): The row label.
        col (str): The column label.
        
    Returns:
        value: The extracted value or pd.NA if not found.
    """
    try:
        value = df.loc[row, col].iloc[0]
    except Exception as e:
        logger.warning(f"Missing data for {row} in {col}: {e}")
        return pd.NA
    return value

def safe_div(numerator, denominator):
    """
    Performs safe division: if numerator or denominator is NA or denominator==0,
    returns pd.NA.
    """
    try:
        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return pd.NA
        return numerator / denominator
    except Exception as e:
        logger.warning(f"Division error: {e}")
        return pd.NA

def compute_ratios(ticker_symbol):
    """
    Computes financial ratios based on fetched financial statements.
    Missing components are replaced with NA.
    
    Parameters:
        ticker_symbol (str): The ticker symbol of the stock.
        
    Returns:
        dict: Dictionary of computed ratios.
    """
    income_stmt, balance_sheet, cash_flow, tick, stock = fetch_financial_data(ticker_symbol)
    results = {}
    try:
        # Profitability
        net_income = extract_value(income_stmt, "Net Income", DEFAULT_YEAR)
        revenue = extract_value(income_stmt, "Total Revenue", DEFAULT_YEAR)
        results["net_profit_margin"] = safe_div(net_income, revenue)

        cogs = extract_value(income_stmt, "Cost Of Revenue", DEFAULT_YEAR)
        revenue_minus_cogs = (revenue - cogs) if (not pd.isna(revenue) and not pd.isna(cogs)) else pd.NA
        results["gross_profit_margin"] = safe_div(revenue_minus_cogs, revenue)

        total_assets = extract_value(balance_sheet, "Total Assets", DEFAULT_YEAR)
        results["roa"] = safe_div(net_income, total_assets)

        stockholders_equity = extract_value(balance_sheet, "Stockholders Equity", DEFAULT_YEAR)
        results["roe"] = safe_div(net_income, stockholders_equity)

        # Liquidity
        current_assets = extract_value(balance_sheet, "Current Assets", DEFAULT_YEAR)
        current_liabilities = extract_value(balance_sheet, "Current Liabilities", DEFAULT_YEAR)
        results["current_ratio"] = safe_div(current_assets, current_liabilities)

        inventory = extract_value(balance_sheet, "Inventory", DEFAULT_YEAR)
        current_minus_inventory = (current_assets - inventory) if (not pd.isna(current_assets) and not pd.isna(inventory)) else pd.NA
        results["quick_ratio"] = safe_div(current_minus_inventory, current_liabilities)

        cash_and_cash_equivalents = extract_value(balance_sheet, "Cash And Cash Equivalents", DEFAULT_YEAR)
        results["cash_ratio"] = safe_div(cash_and_cash_equivalents, current_liabilities)

        # Efficiency (using previous year's data from PRIOR_YEAR)
        inventory_prior = extract_value(balance_sheet, "Inventory", PRIOR_YEAR)
        average_inventory = safe_div((inventory + inventory_prior) if (not pd.isna(inventory) and not pd.isna(inventory_prior)) else pd.NA, 2)
        results["inventory_turnover"] = safe_div(cogs, average_inventory)

        accounts_receivable_current = extract_value(balance_sheet, "Accounts Receivable", DEFAULT_YEAR)
        accounts_receivable_prior = extract_value(balance_sheet, "Accounts Receivable", PRIOR_YEAR)
        average_accounts_receivable = safe_div((accounts_receivable_current + accounts_receivable_prior) if (not pd.isna(accounts_receivable_current) and not pd.isna(accounts_receivable_prior)) else pd.NA, 2)
        results["accounts_receivable_turnover"] = safe_div(revenue, average_accounts_receivable)

        results["asset_turnover"] = safe_div(revenue, total_assets)

        # Market Value
        try:
            market_price_per_share = stock.adj_close[-1]
        except Exception as e:
            logger.warning(f"Market price error for {ticker_symbol}: {e}")
            market_price_per_share = pd.NA
        diluted_eps = extract_value(income_stmt, "Diluted EPS", DEFAULT_YEAR)
        results["pe_ratio"] = safe_div(market_price_per_share, diluted_eps)

        shares_outstanding = tick.info.get('sharesOutstanding', pd.NA)
        if pd.isna(shares_outstanding):
            logger.warning("sharesOutstanding not found in ticker.info")
        book_value_per_share = safe_div(stockholders_equity, shares_outstanding)
        results["pb_ratio"] = safe_div(market_price_per_share, book_value_per_share)

        dividends_paid = extract_value(cash_flow, "Cash Dividends Paid", DEFAULT_YEAR)
        dividends_per_share = safe_div(dividends_paid, shares_outstanding)
        results["dividend_yield"] = safe_div(dividends_per_share, market_price_per_share)

        # Leverage
        total_liabilities = extract_value(balance_sheet, "Total Liabilities Net Minority Interest", DEFAULT_YEAR)
        results["debt_to_equity"] = safe_div(total_liabilities, stockholders_equity)
        results["debt_ratio"] = safe_div(total_liabilities, total_assets)

        ebit = extract_value(income_stmt, "EBIT", DEFAULT_YEAR)
        interest_expense = extract_value(income_stmt, "Interest Expense", DEFAULT_YEAR)
        results["interest_coverage_ratio"] = safe_div(ebit, interest_expense)

    except Exception as e:
        logger.error(f"Error computing ratios for {ticker_symbol}: {e}")
    return results

def process_tickers(stocks):
    def process_ticker(row):
        ticker_symbol = row['Ticker']
        start_time = time.time()
        try:
            ratios = compute_ratios(ticker_symbol)
            elapsed = time.time() - start_time
            ratios['ticker'] = ticker_symbol
            ratios['elapsed'] = elapsed
            print(f"Processing {ticker_symbol}... Success in {elapsed:.2f} sec")
            return ratios, None
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Processing {ticker_symbol}... Failed in {elapsed:.2f} sec: {e}")
            return None, ticker_symbol
    
    results = []
    error_tickers = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_ticker, row): row['Ticker'] for idx, row in stocks.iterrows()}
        for future in as_completed(futures):
            res, error = future.result()
            if res is not None:
                results.append(res)
            if error:
                error_tickers.append(error)
    
    return results, error_tickers



if __name__ == "__main__":
    import sys
    ticker_symbol = sys.argv[1] if len(sys.argv) > 1 else "GOOGL"
    ratios = compute_ratios(ticker_symbol)
    for key, value in ratios.items():
        print(f"{key}: {value}")