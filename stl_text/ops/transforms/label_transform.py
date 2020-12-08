from typing import List

import torch.nn as nn


class LabelTransform(nn.Module):

    def __init__(self, label_names: List[str] = None):
        super().__init__()
        self.label_vocab = {}
        for i, label_name in enumerate(label_names):
            self.label_vocab[label_name] = i

    def forward(self, label_name: str) -> int:
        return self.label_vocab[label_name]

