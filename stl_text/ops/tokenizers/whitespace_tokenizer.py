import torch.nn as nn
from typing import Dict, List


class WhitespaceTokenizer(nn.Module):
    def __init__(self):
        super(WhitespaceTokenizer, self).__init__()
        self.vocab: Dict[str, int] = {}

    def forward(self, text: str) -> List[int]:
        return self.numberize(self.tokenize(text))

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def numberize(self, tokens: List[str]) -> List[int]:
        token_ids: List[int] = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(len(self.vocab))
                self.vocab[token] = len(self.vocab)
        return token_ids
