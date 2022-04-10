import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import T_co
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


class MFTCDataset(Dataset):
    def __init__(self, train_data: pd.DataFrame, tokenizer):
        self.train_data = train_data
        self.tokenizer = tokenizer
        self.encodings = self.tokenizer(self.train_data[1].to_list(),
                                        truncation=True, padding=True, max_length=512,
                                        return_tensors='pt')

        self.labels = torch.tensor(self.train_data[range(2, 13)].to_numpy())

    def __getitem__(self, index) -> T_co:
        item = {key: val[index] for key, val in self.encodings.items()}
        item['labels'] = self.labels[index]
        return item

    def __len__(self):
        return len(self.labels)

    def to_device(self, device):
        for item in self.encodings:
            self.encodings[item] = self.encodings[item].to(device)
        self.labels = self.labels.to(device)


def evaluate(model, test_dataset):
    model.eval()
    y_predicted = []
    y_true = []
    test_loader = DataLoader(test_dataset, batch_size=32)
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            predictions = (logits > 0).float().cpu().numpy()
            y_predicted.extend(predictions)
            y_true.extend(batch['labels'].cpu().numpy())

    result = classification_report(y_true, y_predicted, target_names=LABEL_NAMES, output_dict=True)
    print(classification_report(y_true, y_predicted, target_names=LABEL_NAMES))
    return result


# Set the seed to ensure (to some extent) reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

LABEL_NAMES = ["fairness", "non-moral", "purity", "degradation", "loyalty", "care", "cheating", "betrayal",
               "subversion", "authority", "harm"]

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=11)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model.to(device)
train_dataset = MFTCDataset(pd.read_csv("data/mftc/train.csv", header=None), tokenizer)
test_dataset = MFTCDataset(pd.read_csv("data/mftc/test.csv", header=None), tokenizer)

train_dataset.to_device(device)
test_dataset.to_device(device)

print("Results on test set before training")
evaluate(model, test_dataset)

optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_function = torch.nn.BCEWithLogitsLoss()

epochs = 3

for epoch in tqdm(range(epochs)):
    train_loader: DataLoader = DataLoader(train_dataset, batch_size=32)
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'].float()
        output = model(input_ids, attention_mask=attention_mask)
        loss = loss_function(output.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

print("Results on test set after training")
evaluate(model, test_dataset)
