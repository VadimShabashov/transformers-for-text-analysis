from transformers import BertTokenizer, BertForSequenceClassification

import pytorch_lightning as pl
import torch
import torch.nn as nn


class SentimentAnalysisModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model_name = "bert-base-uncased"
        self.num_classes = 6
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.max_len = 256
        self.softmax = nn.Softmax()

    def forward(self, x):
        encoding = self.tokenizer(x,
                                  return_tensors="pt",
                                  truncation=True,
                                  add_special_tokens=True,
                                  max_length=self.max_len,
                                  return_token_type_ids=False,
                                  pad_to_max_length=True,
                                  return_attention_mask=True).to(self.device)
        return self.model(**encoding)

    def predict_step(self, batch, batch_idx):
        logits = self.forward(batch).logits.to(self.device)
        probs = self.softmax(logits)
        return torch.argmax(probs, dim=1)


def load_model(path="sentiment_analysis.ckpt"):
    return SentimentAnalysisModel.load_from_checkpoint(path)
