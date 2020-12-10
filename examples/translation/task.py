from typing import List, Optional

import torch
import torch.nn as nn

from pytorch_lightning.core.lightning import LightningModule
from stl_text.datamodule import TranslationDataModule


class TranslationTask(LightningModule):

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        # TODO is there a more elegant way to do this?
        datamodule: TranslationDataModule,
        eval_bleu: bool = False,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.datamodule = datamodule
        self.eval_bleu = eval_bleu

        if eval_bleu:
            raise NotImplementedError

        self.loss = nn.CrossEntropyLoss(ignore_index=datamodule.target_pad_idx)

    def forward(self, source_text_batch: List[str]) -> torch.Tensor:
        data = self.data_module(source_text_batch)
        for batch in data:
            logits = self.model(batch, batch_idx=None)[0]
            return logits
        # TODO should decoding logic go here or elsewhere?
        #raise NotImplementedError

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        model_output = self.model(batch, batch_idx=batch_idx)
        loss = self.loss(model_output[0], batch["target_token_ids"])
        self.log("loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)
