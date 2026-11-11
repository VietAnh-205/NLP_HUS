import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class IMDbTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes=2, dropout=0.1):
        super(BertClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, features):
        x = self.dropout(features)
        logits = self.classifier(x)
        return logits

class BertForSentimentClassification(nn.Module):
    def __init__(self, config_or_path='bert-base-uncased', num_classes=2, freeze_encoder=False, dropout=0.1):
        super(BertForSentimentClassification, self).__init__()
        
        if isinstance(config_or_path, str):
            config = BertConfig.from_pretrained(config_or_path, local_files_only=True)
            self.bert = BertModel(config)
        else:
            self.bert = BertModel(config_or_path)
            
        self.classifier_head = BertClassificationHead(
            hidden_size=self.bert.config.hidden_size,
            num_classes=num_classes,
            dropout=dropout
        )
        
        if freeze_encoder:
            self.set_encoder_trainable(False)

    def set_encoder_trainable(self, trainable=True):
        for param in self.bert.parameters():
            param.requires_grad = trainable

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier_head(cls_rep)
        return logits

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

if __name__ == '__main__':
    tok = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    sample_texts = ["This movie was phenomenal!", "Terrible plot and boring acting."]
    sample_labels = [1, 0]
    
    dataset = IMDbTextDataset(sample_texts, sample_labels, tok, max_len=16)
    loader = DataLoader(dataset, batch_size=2)
    
    model = BertForSentimentClassification('bert-base-uncased', num_classes=2, freeze_encoder=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    results = evaluate_model(model, loader, device)
    print("Inference Test Accuracy:", results['accuracy'])
    print("Inference Test F1      :", results['f1'])
    print("Probabilities Shape    :", len(results['probabilities']))
