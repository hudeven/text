import torch.nn as nn
from typing import Dict, List, Optional
from iopath.common.file_io import PathManager


class WhitespaceTokenizer(nn.Module):
    def __init__(self, vocab_path: Optional[str] = None, trainable=False, speed: int = 0):
        super(WhitespaceTokenizer, self).__init__()
        self.trainable = trainable
        self.speed = speed  # mock a real tokenizer: slowing down tokenization speed

        self.unknown = "unknown"
        self.vocab: Dict[str, int] = {self.unknown: 0}

        # load vocab
        path_manager = PathManager()
        if vocab_path:
            with path_manager.open(vocab_path, "r") as f:
                for line in f.readlines():
                    token = line.split()[0]
                    self.vocab[token] = len(self.vocab)

    def forward(self, text: str) -> List[int]:
        return self.numberize(self.tokenize(text))

    def tokenize(self, text: str) -> List[str]:
        # slowing down tokenization speed
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
                    token_ids.append(self.vocab[self.unknown])
        return token_ids
