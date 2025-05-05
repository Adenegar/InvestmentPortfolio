import json
import os
from datetime import datetime
# Try importing pandas, but make it optional unless absolutely needed
import pandas as pd
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas library not found. Monte Carlo result retrieval as Series will not be available.")

def save_portfolio_results(portfolio_name, results, file_path=None, simulation="bootstrap"):
    """
    Save portfolio results to a JSON file, organizing data by a detailed portfolio name.
    
    Parameters:
        portfolio_name (str): Unique name of the portfolio incorporating policy, stocks, duration, etc.
        results (list or dict): Results data to save. For Monte Carlo, often a list of returns.
                                For Bootstrap, a list of dictionaries.
        file_path (str, optional): Path to the JSON file. If None, defaults based on simulation type.
        simulation (str): Simulation type, e.g. "bootstrap" or "monte_carlo". Default is "bootstrap".
    """
    if file_path is None:
        # Define default file paths (adjust '../data/' if needed)
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data') # Assumes tools.py is in a 'tools' subdir
        if not os.path.exists(base_dir):
             base_dir = "./data" # Fallback to current directory subfolder 'data'
             print(f"Warning: Default directory '../data/' not found relative to tools.py. Using '{base_dir}'.")

        file_path = os.path.join(base_dir, "monte_carlo_results.json") if simulation == "monte_carlo" \
                    else os.path.join(base_dir, "bootstrap_results.json")
    
    # Create directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {os.path.dirname(file_path)}: {e}")
        # Optional: Fallback to current directory if directory creation fails?
        # file_path = os.path.basename(file_path)
        # print(f"Attempting to save in current directory as {file_path}")
        # No, better to fail clearly if directory isn't writable/creatable.

    # Initialize the data structure
    all_portfolios = {}
    
    # Read existing data if file exists
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0: # Check size > 0
        try:
            with open(file_path, "r") as file:
                all_portfolios = json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: File {file_path} exists but is empty or contains invalid JSON. Starting fresh.")
            all_portfolios = {}
        except Exception as e:
            print(f"Error reading existing results file {file_path}: {e}. Starting fresh.")
            all_portfolios = {} # Avoid overwriting potentially recoverable data? Maybe backup first? For now, overwrite.
            
    # Add timestamp and simulation metadata to results
    # Note: The 'results' variable itself should be the primary data (list or dict)
    results_with_meta = {
        "data": results, # Store the actual results list/dict here
        "simulation_type": simulation,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Update data for this specific portfolio run (using the detailed name as the key)
    all_portfolios[portfolio_name] = results_with_meta
    
    # Write updated data back to file
    try:
        with open(file_path, "w") as file:
            json.dump(all_portfolios, file, indent=4, default=str) # Use default=str for non-serializable types like Timestamps if they sneak in
        print(f"Results for portfolio '{portfolio_name}' saved to {file_path}")
    except IOError as e:
         print(f"Error writing results to {file_path}: {e}")
    except TypeError as e:
         print(f"Error serializing results to JSON for {portfolio_name}. Ensure data is JSON compatible: {e}")


def retrieve_portfolio_results(portfolio_name=None, file_path=None, simulation="bootstrap", data_only=True):
    """
    Retrieve portfolio results from a JSON file using the detailed portfolio name.
    
    Parameters:
        portfolio_name (str, optional): Unique name of the portfolio run to retrieve.
                                       If None, returns all portfolios from the file.
        file_path (str, optional): Path to the JSON file. If None, defaults based on simulation type.
        simulation (str): Simulation type ('bootstrap' or 'monte_carlo') to determine the default file.
        data_only (bool): If True, returns just the 'data' part. If False, returns 
                          the full entry including metadata.
    
    Returns:
        dict or list or pd.Series/pd.DataFrame: The requested portfolio data, or all portfolios.
              Returns None if the file or specific portfolio doesn't exist.
              Returns specific types based on data_only and simulation type (and pandas availability).
    """
    if file_path is None:
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        if not os.path.exists(base_dir):
             base_dir = "./data"
        file_path = os.path.join(base_dir, "monte_carlo_results.json") if simulation == "monte_carlo" \
                    else os.path.join(base_dir, "bootstrap_results.json")
        
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Warning: Results file {file_path} not found.")
        return None
    
    try:
        # Load all portfolios from the JSON file
        with open(file_path, "r") as file:
            all_portfolios = json.load(file)
        
        # Return data for a specific portfolio run if requested
        if portfolio_name is not None:
            if portfolio_name not in all_portfolios:
                print(f"Warning: Portfolio run '{portfolio_name}' not found in {file_path}.")
                return None
            
            portfolio_entry = all_portfolios[portfolio_name]
            
            if data_only:
                result_data = portfolio_entry.get("data") # Safely get data
                # Try converting to pandas objects if appropriate and available
                if PANDAS_AVAILABLE:
                    sim_type = portfolio_entry.get("simulation_type", simulation) # Use stored sim type if available
                    if sim_type == "monte_carlo" and isinstance(result_data, list):
                         try:
                             return pd.Series(result_data)
                         except Exception as e:
                              print(f"Warning: Could not convert Monte Carlo data to Series: {e}. Returning raw list.")
                              return result_data
                    elif sim_type == "bootstrap" and isinstance(result_data, list) and all(isinstance(item, dict) for item in result_data):
                         try:
                             return pd.DataFrame(result_data)
                         except Exception as e:
                              print(f"Warning: Could not convert Bootstrap data to DataFrame: {e}. Returning raw list of dicts.")
                              return result_data
                    else: # Return raw data if not MC list or Bootstrap list-of-dicts
                        return result_data
                else: # Pandas not available
                    return result_data
            else: # Return full entry with metadata
                return portfolio_entry
        
        # Otherwise return all portfolios stored in the file
        else:
            if data_only:
                processed_portfolios = {}
                for name, entry in all_portfolios.items():
                    result_data = entry.get("data")
                    if PANDAS_AVAILABLE:
                        sim_type = entry.get("simulation_type", simulation) # Default to function arg if not stored
                        if sim_type == "monte_carlo" and isinstance(result_data, list):
                             try:
                                 processed_portfolios[name] = pd.Series(result_data)
                             except Exception:
                                 processed_portfolios[name] = result_data # Fallback
                        elif sim_type == "bootstrap" and isinstance(result_data, list) and all(isinstance(item, dict) for item in result_data):
                            try:
                                processed_portfolios[name] = pd.DataFrame(result_data)
                            except Exception:
                                processed_portfolios[name] = result_data # Fallback
                        else:
                            processed_portfolios[name] = result_data
                    else: # Pandas not available
                         processed_portfolios[name] = result_data
                return processed_portfolios
            else: # Return all full entries
                return all_portfolios
                
    except json.JSONDecodeError:
        print(f"Error: Could not parse {file_path} as valid JSON.")
        return None
    except FileNotFoundError: # Should be caught earlier, but belt-and-suspenders
         print(f"Error: File {file_path} not found during retrieval attempt.")
         return None
    except Exception as e:
        print(f"Error retrieving portfolio results from {file_path}: {e}")
        return None