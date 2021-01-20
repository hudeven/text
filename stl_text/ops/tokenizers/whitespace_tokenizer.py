import torch.nn as nn
from typing import Dict, List, Optional
from collections import OrderedDict
from iopath.common.file_io import PathManager
from torchtext.experimental.vocab import vocab

class WhitespaceTokenizer(nn.Module):
    def __init__(self, vocab_path: Optional[str] = None, trainable: bool = False):
        super(WhitespaceTokenizer, self).__init__()
        self.trainable = trainable

        # load vocab
        ordered_dict = OrderedDict()
        if vocab_path:
            path_manager = PathManager()
            with path_manager.open(vocab_path, "r") as f:
                for line in f.readlines():
                    token = line.split()[0]
                    ordered_dict[token] = 1
 
        self.vocab = vocab(ordered_dict)
        self.vocab = self.vocab.to_ivalue()


    def forward(self, text: str) -> List[int]:
        return self.numberize(self.tokenize(text))

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def numberize(self, tokens: List[str]) -> List[int]:
        token_ids = self.vocab(tokens)

        if self.trainable:
            unk_id = self.vocab.__getitem__('<unk>')

            for i,(token,token_id) in enumerate(zip(tokens,token_ids)): 
                if token_id==unk_id:
                    self.vocab.append_token(tokens[i])
                    token_ids[i] = self.vocab.__getitem__(token)

        return token_ids

