#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional

import torch
import torch.nn as nn
from stl_text.ops.modules import MlpDecoder, RobertaEncoder


class RobertaModel(nn.Module):
    def __init__(
            self,
            model_path: Optional[str] = None,
            vocab_size: int = 50265,
            embedding_dim: int = 768,
            num_attention_heads: int = 12,
            num_encoder_layers: int = 12,
            output_dropout: float = 0.4,
            dense_dim: int = 0,
            out_dim: int = 2,
            bias: bool = True,
    ):
        """
        Roberta Model with flatten args.
        To reduce the layers of config, encoder/decoder's args are
        exposed here.

        Args:

            model_path: pre-trained model path for encoder
            vocab_size: vocabulary size for encoder
            embedding_dim: embedding dimension for encoder
            num_attention_heads: num of attention heads for encoder
            num_encoder_layers: num of encoder layers
            output_dropout: dropout for encoder
            dense_dim: dense feature dimension
            out_dim: output dimension for decoder
            bias: bias for decoder
        """
        super().__init__()
        self.encoder = RobertaEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_attention_heads=num_attention_heads,
            num_encoder_layers=num_encoder_layers,
            output_dropout=output_dropout,
            model_path=model_path,
        )
        self.decoder = MlpDecoder(
            in_dim=embedding_dim + dense_dim,
            out_dim=out_dim,
            bias=bias,
            activation="relu",
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        representation = self.encoder(token_ids)
        return self.decoder(representation)
