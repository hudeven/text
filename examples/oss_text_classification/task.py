import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule


class TextClassificationTask(LightningModule):
    def __init__(self, vocab_size, embed_dim, num_class, learning_rate):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.lr = learning_rate
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        labels, texts, offsets = batch
        predited_label = self(texts, offsets)
        loss = torch.nn.functional.cross_entropy(predited_label, labels)
        return loss

    def _eval_step(self, batch, batch_idx):
        labels, texts, offsets = batch
        predited_labels = self(texts, offsets)
        return (predited_labels, labels)

    def _eval_epoch_end(self, outputs):
        total_acc, total_count = 0, 0
        for i, (predited_labels, target_labels) in enumerate(outputs):
            total_acc += (predited_labels.argmax(1) == target_labels).sum().item()
            total_count += predited_labels.size(0)
        return total_acc, total_count
        self.log('val_acc', total_acc/total_count, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def validation_epoch_end(self, valid_outputs):
        total_acc, total_count = self._eval_epoch_end(valid_outputs)
        self.log('val_acc', total_acc/total_count, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, test_outputs):
        total_acc, total_count = self._eval_epoch_end(test_outputs)
        self.log('test_acc', total_acc/total_count, prog_bar=True)
