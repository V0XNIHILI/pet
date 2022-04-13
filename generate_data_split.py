import math
import pandas as pd
import numpy as np
import argparse

# Set the seed to ensure (to some extent) reproducibility
SEED = 42
np.random.seed(SEED)
domains = ["ALM", "Baltimore", "BLM", "Davidson", "Election", "MeToo", "Sandy"]

parser = argparse.ArgumentParser(description="CLI for splitting the MFTC corpus into train/test/unlabeled sets")

parser.add_argument("--path", default=None, type=str, required=True,
                    help="The location of the data dir. Should contain the following files: ALM.csv, Baltimore.csv, "
                         "BLM.csv, Davidson.csv, Election.csv, MeToo.csv, Sandy.csv")

parser.add_argument("--train-split-percentage", default=0.8, type=float, required=False,
                    help="The percentage of data used for training.")

parser.add_argument("--test-split-percentage", default=0.2, type=float, required=False,
                    help="The percentage of data used for testing.")

args = parser.parse_args()

train_percentage = args.train_split_percentage
test_percentage = args.test_split_percentage

assert train_percentage + test_percentage <= 1, "The specified percentages' sum is > 1"

train_data = []
test_data = []
unlabeled_data = []

for domain in domains:
    df = pd.read_csv(f"{args.path}/{domain}.csv")
    train_index = math.floor(len(df) * train_percentage)
    test_index = math.floor(len(df) * (1 - test_percentage))
    train, unlabeled, test = np.split(df, [train_index, test_index])
    train_data.append(train)
    test_data.append(test)
    unlabeled_data.append(unlabeled)

train_data = pd.concat(train_data)
test_data = pd.concat(test_data)
unlabeled_data = pd.concat(unlabeled_data)

train_data.to_csv("data/mftc/train.csv", header=False, index=False)
test_data.to_csv("data/mftc/test.csv", header=False, index=False)
unlabeled_data.to_csv("data/mftc/unlabeled.csv", header=False, index=False)
