import torch
import torch.nn as nn
from typing import Dict, List, Optional, Iterable
from collections import OrderedDict
from iopath.common.file_io import PathManager
from torchtext.experimental.vocab import (
    build_vocab_from_iterator,
    load_vocab_from_file,
    Vocab,
    vocab,
)

class WhitespaceTokenizer(nn.Module):
    r"""White space tokenizer and vocabulary builder

    Args:
        vocab_path Optional[str]: Path to vocabulary file. The file should contain tokens seperated by new line.
        trainable (bool): Indicates whether to add tokens to dictionary on-the-fly

    Example:

        >>> from stl_text.ops.tokenizers import WhitespaceTokenizer
        >>> whitespacetokenizer = WhitespaceTokenizer()
        >>> jit_whitespacetokenizer = torch.jit.script(whitespacetokenizer)
        >>> print(jit_whitespacetokenizer.code)
        def forward(self,
            text: str) -> List[int]:
        _0 = (self.vocab).forward(torch.split(text, None, -1), )
        return _0
        
    """

    def __init__(self, vocab_path: Optional[str] = None):
        super(WhitespaceTokenizer, self).__init__()
        self.unk_token = '<unk>'
        if vocab_path:
            path_manager = PathManager()
            with path_manager.open(vocab_path, "r",encoding='utf-8') as f:
                self.vocab = load_vocab_from_file(f,unk_token=self.unk_token)
        else:
            self.vocab = vocab(OrderedDict(),unk_token=self.unk_token)
            
        self.vocab = self.vocab.to_ivalue() #Remove to_ivalue() PR: https://github.com/pytorch/text/pull/1080


    def forward(self, text: str) -> List[int]:
        return self.vocab(text.split())

    @torch.jit.ignore
    def build_vocab(self, data: List[str]) -> None:
        self.vocab = build_vocab_from_iterator(map(lambda x:x.split(),data))
