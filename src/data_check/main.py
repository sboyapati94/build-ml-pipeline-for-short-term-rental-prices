# steps/data_check/main.py
import argparse
import pandas as pd
import wandb
from mlops_utils import test_data  # Your test logic

def go(args):
    run = wandb.init(job_type="data_check")
    
    # Download artifacts from W&B
    artifact = run.use_artifact(args.csv_artifact, type='dataset')
    csv_path = artifact.download()

    ref_artifact = run.use_artifact(args.reference_artifact, type='dataset')
    ref_path = ref_artifact.download()

    # Load datasets
    df = pd.read_csv(f"{csv_path}/clean_sample.csv")
    ref_df = pd.read_csv(f"{ref_path}/clean_sample.csv")

    # Run tests
    test_data.test_column_names(df)
    test_data.test_class_names(df)
    test_data.test_neighborhood_group(df)
    test_data.test_row_count(df)
    test_data.test_price_range(df, args.min_price, args.max_price)
    test_data.test_kl_divergence(df, ref_df, threshold=args.kl_threshold)

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_artifact", type=str, required=True)
    parser.add_argument("--reference_artifact", type=str, required=True)
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)
    parser.add_argument("--kl_threshold", type=float, required=True)

    args = parser.parse_args()
    go(args)
