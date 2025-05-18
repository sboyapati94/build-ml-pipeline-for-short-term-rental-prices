import argparse
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def go(args):
    #run = wandb.init(project="nyc_airbnb", group="data_cleaning", job_type="data_cleaning", save_code=True)
    run = wandb.init(project="nyc_airbnb", group="eda", job_type="eda", save_code=True)
    run.config.update(args)

    logger.info("Downloading artifact from W&B")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    logger.info("Reading dataset")
    df = pd.read_csv(artifact_path)

    logger.info(f"Filtering rows with price not in [{args.min_price}, {args.max_price}]")
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info("Dropping rows with missing values")
    df.dropna(inplace=True)

    logger.info("Saving cleaned dataset to clean_sample.csv")
    df.to_csv("clean_sample.csv", index=False)

    logger.info("Logging cleaned dataset to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the dataset")

    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)
    parser.add_argument("--output_type", type=str, required=True)
    parser.add_argument("--output_description", type=str, required=True)
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)

    args = parser.parse_args()
    go(args)
