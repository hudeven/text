from pytorch_lightning.core.lightning import LightningModule
import torch
from torch.nn.functional import cross_entropy


class QuestionAnswerTask(LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.lr = learning_rate

    def forward(self, seq_input, tok_type):
        start_pos, end_pos = self.model(seq_input, token_type_input=tok_type)
        return start_pos, end_pos

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        seq_input, target_start_pos, target_end_pos, tok_type = batch
        start_pos, end_pos = self(seq_input, tok_type)
        loss = (cross_entropy(start_pos, target_start_pos) + cross_entropy(end_pos, target_end_pos)) / 2
        return loss

    def validation_step(self, batch, batch_idx):
        seq_input, target_start_pos, target_end_pos, tok_type = batch
        start_pos, end_pos = self(seq_input, tok_type)
        loss = (cross_entropy(start_pos, target_start_pos) + cross_entropy(end_pos, target_end_pos)) / 2
        return loss

    def validation_epoch_end(self, valid_outputs):
        total_loss = 0.0
        for i, loss in enumerate(valid_outputs):
            total_loss += loss.item()
        self.log('val_acc', total_loss/len(valid_outputs), prog_bar=True)
