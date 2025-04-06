import json
import os
from datetime import datetime

def save_portfolio_results(portfolio_name, results, file_path=None, simulation="bootstrap"):
    """
    Save portfolio results to a JSON file, organizing data by portfolio name.
    
    Parameters:
        portfolio_name (str): Name of the portfolio
        results (dict or pandas.Series): Results data to save
        file_path (str, optional): Path to the JSON file. If None, defaults to a file based on simulation type.
        simulation (str): Simulation type, e.g. "bootstrap" or "monte_carlo". Default is "bootstrap".
    """
    if file_path is None:
        file_path = "../data/monte_carlo_results.json" if simulation == "monte_carlo" else "../data/bootstrap_results.json"
    
    # For monte_carlo simulation, if results is a pandas Series (returns_5y), convert to dict
    if simulation == "monte_carlo":
        try:
            import pandas as pd
            if isinstance(results, pd.Series):
                results = results.to_dict()
        except ImportError:
            print("pandas is required to process monte_carlo simulation data.")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Initialize the data structure
    all_portfolios = {}
    
    # Read existing data if file exists
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as file:
                all_portfolios = json.load(file)
        except json.JSONDecodeError:
            # If file exists but is empty or invalid, start with empty dict
            all_portfolios = {}
    
    # Add timestamp to results
    results_with_meta = {
        "data": results,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Update data for this portfolio
    all_portfolios[portfolio_name] = results_with_meta
    
    # Write updated data back to file
    with open(file_path, "w") as file:
        json.dump(all_portfolios, file, indent=4)
    
    print(f"Results for portfolio '{portfolio_name}' saved to {file_path}")

def retrieve_portfolio_results(portfolio_name=None, file_path=None, simulation="bootstrap", data_only=True):
    """
    Retrieve portfolio results from a JSON file.
    
    Parameters:
        portfolio_name (str, optional): Name of the portfolio to retrieve.
                                       If None, returns all portfolios.
        file_path (str, optional): Path to the JSON file. If None, defaults to a file based on simulation type.
        simulation (str): Simulation type, e.g. "bootstrap" or "monte_carlo". Default is "bootstrap".
        data_only (bool): If True, returns just the data. If False, returns 
                          the full entry including metadata like last_updated.
    
    Returns:
        dict: The requested portfolio data, or all portfolios if no name specified.
              Returns None if the file or portfolio doesn't exist.
    """
    if file_path is None:
        file_path = "../data/monte_carlo_results.json" if simulation == "monte_carlo" else "../data/bootstrap_results.json"
        
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Warning: Results file {file_path} not found.")
        return None
    
    try:
        # Load all portfolios
        with open(file_path, "r") as file:
            all_portfolios = json.load(file)
        
        # Return data for a specific portfolio if requested
        if portfolio_name is not None:
            if portfolio_name not in all_portfolios:
                print(f"Warning: Portfolio '{portfolio_name}' not found in {file_path}.")
                return None
            
            # Return just the data or the full entry
            if data_only:
                result = all_portfolios[portfolio_name]["data"]
                if simulation == "monte_carlo":
                    try:
                        import pandas as pd
                        return pd.Series(result)
                    except ImportError:
                        print("pandas is required to convert to pd.Series for monte_carlo simulation.")
                        return result
                else:
                    return result
            else:
                return all_portfolios[portfolio_name]
        
        # Otherwise return all portfolios
        else:
            if data_only:
                if simulation == "monte_carlo":
                    try:
                        import pandas as pd
                        return {name: pd.Series(entry["data"]) for name, entry in all_portfolios.items()}
                    except ImportError:
                        print("pandas is required to convert to pd.Series for monte_carlo simulation.")
                        return {name: entry["data"] for name, entry in all_portfolios.items()}
                else:
                    return {name: entry["data"] for name, entry in all_portfolios.items()}
            else:
                return all_portfolios
                
    except json.JSONDecodeError:
        print(f"Error: Could not parse {file_path} as valid JSON.")
        return None
    except Exception as e:
        print(f"Error retrieving portfolio results: {e}")
        return None
