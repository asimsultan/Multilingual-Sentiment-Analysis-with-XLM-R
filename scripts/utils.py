
import torch
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_text(tokenizer, texts, max_length):
    return tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
