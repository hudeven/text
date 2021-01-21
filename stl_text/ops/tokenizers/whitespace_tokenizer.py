import torch
import torch.nn as nn
from typing import Dict, List, Optional
from collections import OrderedDict
from iopath.common.file_io import PathManager
from torchtext.experimental.vocab import (
    load_vocab_from_file,
    Vocab,
)
from torchtext._torchtext import Vocab as VocabPybind 

class WhitespaceTokenizer(nn.Module): 
    def __init__(self, vocab_path: Optional[str] = None, trainable: bool = False):
        super(WhitespaceTokenizer, self).__init__()
        self.trainable = trainable
        self.unk_token = '<unk>'

        if vocab_path:
            path_manager = PathManager()
            with path_manager.open(vocab_path, "r",encoding='utf-8') as f:
                self.vocab = load_vocab_from_file(f,unk_token=self.unk_token)
        else:
            self.vocab = Vocab(VocabPybind([self.unk_token],self.unk_token))

        self.vocab = self.vocab.to_ivalue() #Remove to_ivalue() PR: https://github.com/pytorch/text/pull/1080


    def forward(self, text: str) -> List[int]:
        return self.numberize(text.split())

    def numberize(self, tokens: List[str]) -> List[int]:
        token_ids = self.vocab(tokens)
        if self.trainable:
            unk_id = self.vocab([self.unk_token])[0]

            for i,(token,token_id) in enumerate(zip(tokens,token_ids)): 
                if token_id==unk_id:
                    self.vocab.append_token(token)
                    token_ids[i] = self.vocab([token])[0]

        return token_ids

