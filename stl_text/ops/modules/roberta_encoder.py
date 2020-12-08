#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional

import torch
import torch.nn as nn
from pytext.models.representations.transformer import (
    MultiheadSelfAttention,
    Transformer,
    TransformerLayer,
)
from pytext.models.representations.transformer.sentence_encoder import (
    translate_roberta_state_dict,
)
from pytext.utils.file_io import PathManager
from torch.serialization import default_restore_location


class RobertaEncoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            num_attention_heads: int,
            num_encoder_layers: int,
            output_dropout: float,
            model_path: Optional[str] = None,
    ):
        super().__init__()
        self.transformer = Transformer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            layers=[
                TransformerLayer(
                    embedding_dim=embedding_dim,
                    attention=MultiheadSelfAttention(
                        embedding_dim, num_attention_heads
                    ),
                )
                for _ in range(num_encoder_layers)
            ],
        )
        self.output_dropout = nn.Dropout(output_dropout)

        self.apply(init_params)
        if model_path:
            with PathManager.open(model_path, "rb") as f:
                roberta_state = torch.load(
                    f, map_location=lambda s, l: default_restore_location(s, "cpu")
                )
                if "model" in roberta_state:
                    roberta_state = translate_roberta_state_dict(roberta_state["model"])
                self.load_state_dict(roberta_state)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        all_layers = self.transformer(tokens)  # lists of T x B x C
        last_layer = all_layers[-1].transpose(0, 1)
        sentence_rep = last_layer[:, 0, :]
        return self.output_dropout(sentence_rep)


def init_params(module):
    """Initialize the RoBERTa weights for pre-training from scratch."""

    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
