#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional

import torch
import torch.nn as nn

class MlpDecoder(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool,
            hidden_dims: List[int] = None,
            activation: str = "relu",
    ) -> None:
        super().__init__()
        layers = []
        for dim in hidden_dims or []:
            layers.append(nn.Linear(in_dim, dim, bias))
            layers.append(get_activation(activation))
            in_dim = dim
        layers.append(nn.Linear(in_dim, out_dim, bias))

        self.mlp = nn.Sequential(*layers)

    def forward(
            self, representation: torch.Tensor, dense: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if dense is not None:
            representation = torch.cat([representation, dense], 1)
        return self.mlp(representation)


def get_activation(name, dim=1):
    if name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "gelu":
        return nn.GELU()
    elif name == "glu":
        return nn.GLU(dim=dim)
    else:
        raise RuntimeError(f"{name} is not supported")
