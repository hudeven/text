
from pytorch_lightning import LightningModule, LightningDataModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from copy import deepcopy
from torch.optim import AdamW


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
   

    def _encode_sequence(self, token_ids, encoder_model):
        encoded_seq = encoder_model(token_ids) # bs x d
        return encoded_seq


    def _get_mask(self, contexts_ids):
        contexts_mask = torch.greater(contexts_ids.sum(-1), 0).long()
        return contexts_mask

    def forward(self, query_ids, contexts_ids):
        # encode query and contexts
        query_repr = self._encode_sequence(query_ids, self.query_model) # bs x d

        bs, ctx_cnt, ctx_len = contexts_ids.size()
        contexts_ids_flat = torch.reshape(contexts_ids, (-1, ctx_len)) # bs * ctx_cnt x ctx_len
        contexts_repr_flat = self._encode_sequence(contexts_ids_flat, self.context_model) # bs * ctx_cnt x d
        contexts_repr = torch.reshape(contexts_repr_flat, (bs, ctx_cnt, -1)) # bs x ctx_cnt x d

        scores = torch.matmul(query_repr, torch.transpose(contexts_repr, 1, 2)) # bs x ctx_cnt
        
        contexts_mask = self._get_mask(contexts_ids)
        # TO DO: mask the log_softmax
        softmax_scores = F.log_softmax(scores, 1)
        
        return softmax_scores


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        query_ids = batch["query_ids"] # bs x tokens
        contexts_ids = batch["contexts_ids"] # bs x ctx_cnt x ctx_len
        contexts_is_pos = batch["contexts_is_pos"] # bs x ctx_cnt
        
        pred_context_scores = self(query_ids, contexts_ids)
        loss = self.loss(pred_context_scores, contexts_is_pos.long())
        
        return loss

    def _get_correct_predictions_cnt(self, pred_scores, correct, mask):
        """
            Predictions with max score are considered "correct". 
            
        """
        max_score, max_idxs = torch.max(pred_scores, 1)
        correct_predictions = (max_idxs == torch.tensor(correct).to(max_idxs.device)) * mask
        correct_predictions_count = correct_predictions.sum()

        return correct_predictions_count

    def _eval_step(self, batch, batch_idx):
        query_ids = batch["query_ids"] # bs x tokens
        contexts_ids = batch["contexts_ids"] # bs x ctx_cnt x ctx_len
        contexts_is_pos = batch["contexts_is_pos"] # bs x ctx_cnt
        
        pred_context_scores = self(query_ids, contexts_ids)
        loss = self.loss(pred_context_scores, contexts_is_pos.long())
        
        #correct_predictions_count = self._get_correct_predictions_cnt(pred_context_scores, contexts_is_pos, self._get_mask(contexts_ids))
        
        contexts_mask = self._get_mask(contexts_ids)
        return pred_context_scores, contexts_is_pos, contexts_mask

    def _eval_epoch_end(self, outputs):
        total_acc, total_count = 0, 0
        for i, (predited_labels, target_labels, mask) in enumerate(outputs):
            total_acc += ((predited_labels == target_labels) * mask).sum().item() / mask.sum()
            total_count += predited_labels.size(0)
        return total_acc, total_count

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

