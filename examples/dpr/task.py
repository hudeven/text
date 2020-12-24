
from pytorch_lightning import LightningModule, LightningDataModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from copy import deepcopy


# Implementation of https://arxiv.org/abs/2004.04906. 
# Logic and some code from the original https://github.com/facebookresearch/DPR/
class DenseRetrieverTask(LightningModule):
    def __init__(
        self,
        query_model: nn.Module,
        context_model: nn.Module,
        datamodule: LightningDataModule,
        lr: float = 1e-3,
    ):
        super().__init__()
        
        # encoder setup
        self.query_model = query_model
        if context_model is None:
            context_model = deepcopy(query_model)
        else:
            self.context_model = context_model

        self.loss = nn.NLLLoss()
        self.lr = lr
   

    def forward(self, query_ids, contexts_ids, contexts_is_pos):
        print(query_ids, contexts_ids, contexts_is_pos)


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    

    # def init_weights(self):
    #     initrange = 0.5
    #     self.embedding.weight.data.uniform_(-initrange, initrange)
    #     self.fc.weight.data.uniform_(-initrange, initrange)
    #     self.fc.bias.data.zero_()

    # def forward(self, text, offsets):
    #     embedded = self.embedding(text, offsets)
    #     return self.fc(embedded)

    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.5)
    #     return [optimizer], [scheduler]

    # def training_step(self, batch, batch_idx):
    #     labels, texts, offsets = batch
    #     predited_label = self(texts, offsets)
    #     loss = torch.nn.functional.cross_entropy(predited_label, labels)
    #     return loss

    # def _eval_step(self, batch, batch_idx):
    #     labels, texts, offsets = batch
    #     predited_labels = self(texts, offsets)
    #     return (predited_labels, labels)

    # def _eval_epoch_end(self, outputs):
    #     total_acc, total_count = 0, 0
    #     for i, (predited_labels, target_labels) in enumerate(outputs):
    #         total_acc += (predited_labels.argmax(1) == target_labels).sum().item()
    #         total_count += predited_labels.size(0)
    #     return total_acc, total_count

    # def validation_step(self, batch, batch_idx):
    #     return self._eval_step(batch, batch_idx)

    # def validation_epoch_end(self, valid_outputs):
    #     total_acc, total_count = self._eval_epoch_end(valid_outputs)
    #     self.log('val_acc', total_acc/total_count, prog_bar=True)

    # def test_step(self, batch, batch_idx):
    #     return self._eval_step(batch, batch_idx)

    # def test_epoch_end(self, test_outputs):
    #     total_acc, total_count = self._eval_epoch_end(test_outputs)
    #     self.log('test_acc', total_acc/total_count, prog_bar=True)

