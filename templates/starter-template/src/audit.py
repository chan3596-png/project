import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Audit dataset")
    parser.add_argument("--dataset", required=True, help="Dataset identifier (e.g. 05-CMP-A)")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}...")
    
    # Placeholder: In reality you would load a CSV or dataframe here
    # df = pd.read_csv(f"data/{args.dataset}.csv")
    
    # Creating a dummy dataframe for skeleton
    df = pd.DataFrame({'dummy_col': [1, 2, 3]})
    
    print("--- DataFrame Info ---")
    df.info()
    
    print("\n--- DataFrame Describe ---")
    print(df.describe())
    
    print("\n--- Missing Values Count ---")
    print(df.isnull().sum())

if __name__ == "__main__":
    main()
