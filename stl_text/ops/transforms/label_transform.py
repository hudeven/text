from typing import List

import torch.nn as nn


class LabelTransform(nn.Module):

    def __init__(self, label_names: List[str] = None):
        super().__init__()
        self.name_to_idx = {}
        self.id_to_name = []
        for i, label_name in enumerate(label_names):
            self.name_to_idx[label_name] = i
            self.id_to_name.append(label_name)

    def forward(self, label_name: str) -> int:
        return self.encode(label_name)

    def encode(self, label_name: str) -> int:
        return self.name_to_idx[label_name]

    def decode(self, label_idx: int) -> str:
        return self.id_to_name[label_idx]

