import torch
import pytorch_lightning as pl

from transformers import BertTokenizer
from transformers import BertForSequenceClassification


class ToxicClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Get available device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Model and tokenizer
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def forward(self, x):
        tokens = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model(**tokens)[0]
        return outputs

    def predict_step(self, batch, batch_idx):
        output = self.forward(batch)
        return (torch.sigmoid(output) > 0.5).to(torch.int32)


def get_toxic_classifier():
    return ToxicClassifier.load_from_checkpoint("toxic_classifier.ckpt")
