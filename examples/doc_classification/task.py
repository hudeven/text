from typing import Callable, List

import torch
import torch.nn as nn
from stl_text.models import RobertaModel
from stl_text.datamodule import DocClassificationDataModule
from pytorch_lightning import metrics
from torch.optim import AdamW
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from pytorch_lightning import LightningModule


class DocClassificationTask(LightningModule):

    def __init__(self, datamodule: DocClassificationDataModule, num_class: int = 2, lr: float = 0.01):
        super().__init__()
        self.num_class = num_class
        self.lr = lr

        self.text_transform = datamodule.text_transform
        self.model = None
        self.optimizer = None
        self.loss = torch.nn.CrossEntropyLoss()
        self.valid_acc = metrics.Accuracy()
        self.test_acc = metrics.Accuracy()

    def setup(self, stage: str):
        self.model = RobertaModel(
            vocab_size=1000,
            embedding_dim=1000,
            num_attention_heads=1,
            num_encoder_layers=1,
            output_dropout=0.4,
            out_dim=self.num_class,
        )
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

    def forward(self, text_batch: List[str]) -> Tensor:
        token_ids: List[Tensor] = [torch.tensor(self.text_transform(text), dtype=torch.long) for text in text_batch]
        model_inputs: Tensor = pad_sequence(token_ids, batch_first=True)
        return self.model(model_inputs)

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        logits = self.model(batch["token_ids"])
        loss = self.loss(logits, batch["label_id"])
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.model(batch["token_ids"])
        loss = self.loss(logits, batch["label_id"])
        self.valid_acc(logits, batch["label_id"])
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log("checkpoint_on", loss, on_epoch=True, sync_dist=True)
        self.log("valid_acc", self.valid_acc, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        logits = self.model(batch["token_ids"])
        loss = self.loss(logits, batch["label_id"])
        self.test_acc(logits, batch["label_id"])
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)
        self.log("test_acc", self.test_acc, on_epoch=True, sync_dist=True)
