import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SPLIT_DIR = os.path.join(SCRIPT_DIR, 'split_csv_files')

COMBINED_CSV = os.path.join(SCRIPT_DIR, 'processed_tweets_AMZN_2015-01-01_to_2019-12-31.csv')

def combine_csv(split_dir, combined_csv):
    
    df_list = []
    
    split_files = [file for file in os.listdir(split_dir) if file.endswith('.csv')]
    
    if not split_files:
        print(f"No CSV files found in the directory '{split_dir}'.")
        return
    
    split_files.sort()
    
    for file in split_files:
        file_path = os.path.join(split_dir, file)
        print(f"Loading {file_path}...")
        
        df_part = pd.read_csv(file_path)
        
        df_list.append(df_part)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    
    combined_df.to_csv(combined_csv, index=False)
    
    print(f"Combined CSV saved to {combined_csv} with {len(combined_df)} rows.")

if __name__ == "__main__":
    combine_csv(SPLIT_DIR, COMBINED_CSV)
