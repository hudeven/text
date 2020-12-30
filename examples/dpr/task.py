
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

        scores = torch.einsum("bd,bld->bl",query_repr, contexts_repr)
        contexts_mask = self._get_mask(contexts_ids)
        
        softmax_scores = F.log_softmax(self._mask_before_softmax(scores, contexts_mask), 1)
        
        return softmax_scores


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        query_ids = batch["query_ids"] # bs x tokens
        contexts_ids = batch["contexts_ids"] # bs x ctx_cnt x ctx_len
        contexts_is_pos = batch["contexts_is_pos"] # bs x ctx_cnt
        
        pred_context_scores = self(query_ids, contexts_ids)
        loss = self.loss(pred_context_scores, torch.argmax(contexts_is_pos.long(), dim=1))
        
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
        contexts_mask = self._get_mask(contexts_ids)

        return pred_context_scores, contexts_is_pos, contexts_mask

    def _mask_before_softmax(self, value, mask):
        """
            Mask with very negative numbers.
        """
        mask = mask.float()
        masked_value = value * mask + (mask + 1e-45).log()
        return masked_value
        

    def _eval_epoch_end(self, outputs):
        # With this data we dont have the gold rank but many positive and negative contexts. 
        # So we will count what number of positives are in top N

        total_avg_rank, total_count = 0, 0
        for i, (pred_scores, target_labels, mask) in enumerate(outputs):
            values, indices = torch.sort(pred_scores, dim=1, descending=True)
            avg_rank = torch.sum(indices * target_labels, dim=1)
            total_avg_rank += avg_rank.sum()
            total_count += pred_scores.size(0)

        return total_avg_rank, total_count

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def validation_epoch_end(self, valid_outputs):
        total_avg_rank, total_count = self._eval_epoch_end(valid_outputs)
        self.log('total_avg_rank', total_count, prog_bar=True)
        self.log('test_avr_rank', total_avg_rank/total_count, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, test_outputs):
        total_avg_rank, total_count = self._eval_epoch_end(test_outputs)
        self.log('total_avg_rank', total_count, prog_bar=True)
        self.log('test_avr_rank', total_avg_rank/total_count, prog_bar=True)

