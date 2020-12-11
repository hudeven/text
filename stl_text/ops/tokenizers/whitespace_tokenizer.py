import random
import torch.nn as nn
from typing import Dict, List


class WhitespaceTokenizer(nn.Module):
    def __init__(self, trainable=False, speed: int = 0):
        super(WhitespaceTokenizer, self).__init__()
        self.vocab: Dict[str, int] = {}
        self.trainable = trainable
        self.speed = speed # mock a real tokenizer: slowing down tokenization speed

    def forward(self, text: str) -> List[int]:
        return self.numberize(self.tokenize(text))

    def tokenize(self, text: str) -> List[str]:
        count = 0
        for i in range(self.speed):
            count += 1
        return text.split()

    def numberize(self, tokens: List[str]) -> List[int]:
        token_ids: List[int] = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                if self.trainable:
                    token_ids.append(len(self.vocab))
                    self.vocab[token] = len(self.vocab)
                else:
                    token_ids.append(random.randint(0, 100))
        return token_ids
