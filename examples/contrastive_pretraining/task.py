from copy import deepcopy

import hydra
from pytorch_lightning import LightningModule, LightningDataModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


# Implementation of arXiv:2005.12766. Most of the logic is from <https://github.com/facebookresearch/moco>.
class CertTask(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        datamodule: LightningDataModule,
        embedding_dim: int,
        queue_size: int,
        temperature: float = 0.07,
        momentum: float = 0.999,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.queue_size = queue_size
        self.temperature = temperature
        self.momentum = momentum

        self.text_transform = datamodule.text_transform

        # biencoder setup – sample encoder is a frozen clone
        self.anchor_encoder = model
        self.sample_encoder = deepcopy(self.anchor_encoder)
        for param in self.sample_encoder.parameters():
            param.requires_grad = False

        self.optimizer = optimizer
        self.loss = nn.CrossEntropyLoss()

        # queue setup
        self.register_buffer("queue", torch.randn(self.embedding_dim, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def configure_optimizers(self):
        return self.optimizer

    def _update_sample_encoder(self):
        for param, tracked_param in zip(self.sample_encoder.parameters(), self.anchor_encoder.parameters()):
            param.data = (
                self.momentum * param.data + (1 - self.momentum) * tracked_param.data
            )

    def _encode(self, anchor_ids, positive_ids):
        anchors = self.anchor_encoder(anchor_ids)
        anchors = F.normalize(anchors)

        with torch.no_grad():
            self._update_sample_encoder()
            positives = self.sample_encoder(positive_ids)
            positives = F.normalize(positives)

        return anchors, positives

    def forward(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, update_queue=True):
        assert batch["anchor_ids"].size(0) == batch["positive_ids"].size(0)

        anchors, positives = self._encode(batch["anchor_ids"], batch["positive_ids"])
        positive_logits = torch.einsum("ij,ij->i", (anchors, positives)).unsqueeze(-1)
        negative_logits = torch.einsum("ij,jk", (anchors, self.queue.clone().detach()))

        logits = torch.cat((positive_logits, negative_logits), dim=1) / self.temperature
        labels = torch.zeros(positive_logits.size(0), dtype=torch.long)

        if update_queue:
            self._dequeue_and_enqueue(positives)

        loss = self.loss(logits, labels)
        return {"loss": loss, "log": {"loss": loss}}

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, update_queue=False)
        return {"val_loss": loss, "log": {"val_loss": loss}}

    @torch.no_grad()
    def _dequeue_and_enqueue(self, samples):
        samples = (
            samples.clone().detach()
        )  # for multi-gpu replace with: concat_all_gather(samples)
        batch_size = samples.size(0)

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the samples at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = samples.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr = torch.tensor(ptr, dtype=torch.long)
