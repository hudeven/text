
from pytorch_lightning import LightningModule, LightningDataModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

# Implementation of https://arxiv.org/abs/2004.04906. 
# Logic and some code from the original https://github.com/facebookresearch/DPR/
class DenseRetrieverTask(LightningModule):
    def __init__(
        self,
        query_encoder: nn.Module,
        context_encoder: nn.Module,
        datamodule: LightningDataModule,
        optimizer: Optimizer,
    ):
        super().__init__()
        
        # encoder setup
        self.query_encoder = query_encoder
        self.context_encoder = query_encoder

        self.optimizer = Optimizer
