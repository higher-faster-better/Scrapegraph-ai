import os
import pandas as pd
from dotenv import load_dotenv
from scrapegraphai.graphs import CSVScraperGraph
from scrapegraphai.utils import convert_to_csv, convert_to_json, prettify_exec_info

# Load environment variables
load_dotenv()

# ************************************************
# Configuration Constants
# ************************************************

FILE_NAME = "inputs/username.csv"
GRAPH_CONFIG = {
    "llm": {
        "api_key": os.getenv("API_KEY"),  # Store API key securely in the .env file
        "model": "oneapi/qwen-turbo",
        "base_url": "http://127.0.0.1:3000/v1",  # OneAPI URL
    }
}

# ************************************************
# Helper Functions
# ************************************************

def get_file_path(filename):
    """Get the absolute file path for the CSV file."""
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(curr_dir, filename)

def read_csv(file_path):
    """Read the CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def run_scraper(text, config, prompt="List me all the last names"):
    """Run the scraper graph with the given text and configuration."""
    scraper = CSVScraperGraph(prompt=prompt, source=str(text), config=config)
    return scraper.run(), scraper

def save_results(result):
    """Save the result in both CSV and JSON formats."""
    convert_to_csv(result, "result")
    convert_to_json(result, "result")

# ************************************************
# Main Script Execution
# ************************************************

def main():
    # Read CSV file
    file_path = get_file_path(FILE_NAME)
    text = read_csv(file_path)
    
    # Run scraper graph
    result, scraper = run_scraper(text, GRAPH_CONFIG)
    print(result)
    
    # Get and print execution info
    graph_exec_info = scraper.get_execution_info()
    print(prettify_exec_info(graph_exec_info))
    
    # Save results
    save_results(result)

# Run the script
if __name__ == "__main__":
    main()
