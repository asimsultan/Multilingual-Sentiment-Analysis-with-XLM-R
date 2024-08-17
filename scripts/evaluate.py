
import torch
import argparse
import pandas as pd
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from utils import get_device, preprocess_text, SentimentDataset

def main(model_path, data_path):
    # Load Model and Tokenizer
    model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)

    # Device
    device = get_device()
    model.to(device)

    # Load Dataset
    dataset = pd.read_csv(data_path)
    val_texts, val_labels = dataset['text'], dataset['label']
    val_encodings = preprocess_text(tokenizer, val_texts.tolist(), max_length=128)

    # DataLoader
    val_dataset = SentimentDataset(val_encodings, val_labels.tolist())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Evaluation Function
    def evaluate(model, data_loader, device):
        model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    # Evaluate
    avg_loss, accuracy = evaluate(model, val_loader, device)
    print(f'Average Loss: {avg_loss}')
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing validation data')
    args = parser.parse_args()
    main(args.model_path, args.data_path)
