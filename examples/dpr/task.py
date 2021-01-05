
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
        in_batch_train: bool = True,
        in_batch_eval: bool = False,
    ):
        super().__init__()
        
        # encoder setup
        self.query_model = query_model
        if context_model is None:
            context_model = deepcopy(query_model)
        else:
            self.context_model = context_model

        self.in_batch_train = in_batch_train
        self.in_batch_eval = in_batch_eval
        self.loss = nn.NLLLoss()
        self.lr = lr
        
    def _encode_sequence(self, token_ids, encoder_model):
        encoded_seq = encoder_model(token_ids) # bs x d
        return encoded_seq


    def _get_mask(self, contexts_ids):
        contexts_mask = torch.greater(contexts_ids.sum(-1), 0).long()
        return contexts_mask

    def forward(self, query_ids, contexts_ids, in_batch):
        if in_batch:
            return self.forward_in_batch(query_ids, contexts_ids)
        else:
            return self.forward_multi_contexts(query_ids, contexts_ids)

    def forward_multi_contexts(self, query_ids, contexts_ids):
        """
            We use this when each query comes with multiple contexts (1 positive + N negative) 
            and we want to compute the similarity only on per-query basis.
        """
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

    def forward_in_batch(self, query_ids, contexts_ids):
        """
            We use this when each query comes with only 1 context (positive) 
            and the negatives are selected from other contexts in the batch.
        """
        # encode query and contexts
        query_repr = self._encode_sequence(query_ids, self.query_model) # bs x d
        contexts_repr = self._encode_sequence(contexts_ids, self.context_model) # bs x d
        
        scores = torch.matmul(query_repr, contexts_repr.transpose(0,1))
        
        log_softmax_scores = F.softmax(scores, 1)
        
        return log_softmax_scores


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        """
            This receives queries, each with mutliple contexts. 
        """
        query_ids = batch["query_ids"] # bs x tokens
        contexts_ids_multi = batch["contexts_ids"] # bs x ctx_cnt x ctx_len
        contexts_is_pos = batch["contexts_is_pos"] # bs x ctx_cnt
        
        if self.in_batch_train:
            # In this case we only need 1 positive context
            contexts_ids  = contexts_ids_multi[:,0,:] # the first one is the positive
            pred_context_scores = self(query_ids, contexts_ids, in_batch=True) # bs x bs
            positive = torch.arange(0, query_ids.size(0), dtype=torch.long)
            #contexts_mask = torch.ones(positive.size())
            loss = self.loss(torch.log(pred_context_scores), positive)
        else:
            pred_context_scores = self(query_ids, contexts_ids_multi, in_batch=False) # bs x ctx_cnt
            #contexts_mask = self._get_mask(contexts_ids_multi)
            positive = torch.argmax(contexts_is_pos.long(), dim=1)
            loss = self.loss(torch.log(pred_context_scores), positive, dim=1)
        
        # calc ranks for training epoch
        values, indices = torch.sort(pred_context_scores, dim=1, descending=True)
        avg_rank = (indices * F.one_hot(positive)).sum(-1).float().mean()

        self.log("batch_avg_rank", avg_rank, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

   
    # def training_epoch_end(self, train_outputs):
    #     total_avg_rank, total_max_rank, total_count, loss = self._eval_epoch_end(train_outputs)
    #     self.log('train_avg_rank', total_avg_rank/total_count, prog_bar=True)
    #     self.log('total_count', total_count, prog_bar=True)
    #     self.log('max_rank_avg', total_max_rank/total_count, prog_bar=True)
    #     self.log('loss', loss, prog_bar=True)

    def _eval_step(self, batch, batch_idx):
        query_ids = batch["query_ids"] # bs x tokens
        contexts_ids_multi = batch["contexts_ids"] # bs x ctx_cnt x ctx_len
        contexts_is_pos = batch["contexts_is_pos"] # bs x ctx_cnt
        
        if self.in_batch_eval:
            contexts_ids  = contexts_ids_multi[:,0,:] # the first one is the positive
            pred_context_scores = self(query_ids, contexts_ids, in_batch=True) # bs x bs
            contexts_is_pos = F.one_hot(torch.arange(0, query_ids.size(0)))
            contexts_mask = torch.ones(contexts_is_pos.size())
            loss = self.loss(torch.log(pred_context_scores), torch.arange(0, query_ids.size(0)))
        else:
            pred_context_scores = self(query_ids, contexts_ids_multi, in_batch=False)
            contexts_mask = self._get_mask(contexts_ids_multi)
            loss = self.loss(torch.log(pred_context_scores), torch.argmax(contexts_is_pos.long(), dim=1))

        return pred_context_scores, contexts_is_pos, contexts_mask, loss

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

        total_avg_rank, total_max_rank, total_count = 0, 0, 0
        total_loss = 0
        for i, (pred_scores, target_labels, mask, loss) in enumerate(outputs):
            values, indices = torch.sort(pred_scores, dim=1, descending=True)

            # calc the avg rank for positive passages element-wise
            avg_rank = torch.sum(indices * target_labels, dim=1).float() / target_labels.sum(dim=1) 
            total_avg_rank += avg_rank.sum()
            
            # the number of passages is the max rank
            total_max_rank += torch.sum(mask) 
            total_count += pred_scores.size(0)

            total_loss += loss

        return total_avg_rank, total_max_rank, total_count, total_loss/(i+1)

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def validation_epoch_end(self, valid_outputs):
        total_avg_rank, total_max_rank, total_count, loss = self._eval_epoch_end(valid_outputs)
        self.log('valid_avg_rank', total_avg_rank/total_count, prog_bar=True)
        self.log('total_count', total_count, prog_bar=True)
        self.log('max_rank', total_max_rank.float()/total_count, prog_bar=True)
        self.log('loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, test_outputs):
        total_avg_rank, total_max_rank, total_count, loss = self._eval_epoch_end(test_outputs)
        self.log('test_avg_rank', total_avg_rank/total_count, prog_bar=True)
        self.log('total_count', total_count, prog_bar=True)
        self.log('max_rank', total_max_rank/total_count, prog_bar=True)
        self.log('loss', loss, prog_bar=True)

