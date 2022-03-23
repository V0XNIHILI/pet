import pandas as pd
import numpy as np
import argparse
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report
from tqdm import tqdm


def create_train_test_data(split_percentage=0.2):
    train_data = []
    test_data = []
    domains = ["ALM", "Baltimore", "BLM", "Davidson", "Election", "MeToo", "Sandy"]
    for domain in domains:
        df = pd.read_csv(f"data/{domain}.csv")
        test, train, _ = np.split(df, [int(split_percentage * len(df)), len(df)])
        train_data.append(train)
        test_data.append(test)
    return pd.concat(train_data), pd.concat(test_data)


train_data, test_data = create_train_test_data()

label_names = test_data.columns[2:]

if not os.path.exists("data/mftc/train.csv"):
    train_data.to_csv("data/mftc/train.csv")

if not os.path.exists("data/mftc/test.csv"):
    test_data.to_csv("data/mftc/test.csv")

parser = argparse.ArgumentParser(description="CLI for evaluating a model on MFTC")

parser.add_argument("--path", default=None, type=str, required=True,
                    help="The location of the model dir. Should contain a config.json file. If not specified it "
                         "defaults to the base pretrained model")

parser.add_argument("--max-test-size", default=len(test_data), type=int, required=False,
                    help="The number of test points to evaluate on.")

args = parser.parse_args()

loaded_model = AutoModelForSequenceClassification.from_pretrained(args.path)
loaded_model.eval()
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

y_predicted = []
y_true = []

test_data = test_data.to_numpy()[:args.max_test_size]
np.random.shuffle(test_data)

with torch.no_grad():
    for i in tqdm(range(len(test_data))):
        data_point = test_data[i]
        encoded_input = tokenizer(data_point[1], return_tensors='pt')
        output = loaded_model(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])
        y_predicted.append((output.logits.numpy().squeeze() >= 0).astype(np.int64))
        y_true.append(data_point[2:].astype(np.int64))

print(classification_report(np.array(y_true), np.array(y_predicted), target_names=label_names))
