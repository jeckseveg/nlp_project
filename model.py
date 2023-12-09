import tqdm
import torch
import numpy as np
import pandas as pd
import transformers
import pytorch_lightning as pl
from transformers import pipeline
from torch.nn import functional as F
from transformers import RobertaTokenizer#, AdamW#, Trainer, TrainingArguments

class ToxicClassifier(pl.LightningModule):
    """Toxic comment classification for the Jigsaw challenges.
    Args:
    """

    def __init__(self):
        super().__init__()

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = torch.hub.load('unitaryai/detoxify','unbiased_toxic_roberta')

        self.tokenizer = self.tokenizer
        self.model = self.model
        self.bias_loss = False

    def forward(self, x):
        print(type(x), 'TYPE')
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        output = self.model(**inputs)[0]
        return output[:,:6]

    def format_output(self, label, output):
        # logic for different datasets
        if label.shape[1] == 4: # youtube
            output = output[:,[0,2,3,5]]
        return output

    def training_step(self, batch, batch_idx):
        x, label = batch
        output = self.forward(x)
        output = self.format_output(label, output)
        loss = self.binary_cross_entropy(output, label)
        self.log("train_loss", loss, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, label = batch
        output = self.forward(x)
        output = self.format_output(label, output)
        loss = self.binary_cross_entropy(output, label)
        acc = self.binary_accuracy(output, label)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        x, label = batch
        output = self.forward(x)
        output = self.format_output(label, output)
        loss = self.binary_cross_entropy(output, label)
        acc = self.binary_accuracy(output, label)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=2e-5, weight_decay=0.01, eps=1e-8)


    def binary_cross_entropy(self, input, target):
        """Custom binary_cross_entropy function.

        Args:
            output ([torch.tensor]): model predictions
            meta ([dict]): meta dict of tensors including targets and weights

        Returns:
            [torch.tensor]: model loss
        """
        target = target.to(input.device)
        return F.binary_cross_entropy_with_logits(input, target.float())

    def binary_accuracy(self, output, target):
        """Custom binary_accuracy function.

        Args:
            output ([torch.tensor]): model predictions
            meta ([dict]): meta dict of tensors including targets and weights

        Returns:
            [torch.tensor]: model accuracy
        """
        with torch.no_grad():
            mask = target != -1
            pred = torch.sigmoid(output[mask]) >= 0.5
            correct = torch.sum(pred.to(output[mask].device) == target[mask])
            if torch.sum(mask).item() != 0:
                correct = correct.item() / torch.sum(mask).item()
            else:
                correct = 0

        return torch.tensor(correct)