#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from pytext.data.utils import BOS, EOS, MASK, PAD, UNK
from typing import Dict, List, Optional
from iopath.common.file_io import PathManager
from torchtext.experimental.vocab import (
    load_vocab_from_file,
    build_vocab_from_iterator,
    vocab,
)

SPECIAL_TOKEN_REPLACEMENT = {
    "[UNK]": UNK,
    "[PAD]": PAD,
    "[CLS]": BOS,
    "[MASK]": MASK,
    "[SEP]": EOS,
}

class TruncateTransform(nn.Module):
    def __init__(self, max_seq_len: int):
        super().__init__()
        assert max_seq_len > 0
        self.max_seq_len: int = max_seq_len

    def forward(self, token_ids: List[List[int]]) -> List[List[int]]:
        return [token_id[: self.max_seq_len] for token_id in token_ids]


class VocabTransform(nn.Module):
    """
    from stl_text.ops.vocab import VocabTransform
    vocab = VocabTransform(vocab_path=vocab_path)
    jit_vocab = torch.jit.script(vocab)
    print(jit_vocab.code)

    >>>>>>>>>>>>>>>>>>>>>>>>>>OUTPUT of JIT code<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def forward(self,
        tokens: List[List[str]]) -> List[List[int]]:
    tokens_idx = annotate(List[List[int]], [])
    for _0 in range(torch.len(tokens)):
        token = tokens[_0]
        _1 = torch.append(tokens_idx, (self.vocab).forward(token, ))
    tokens_idx0 = (self.truncate_transform).forward(tokens_idx, )
    if self.add_bos:
        tokens_idx2 = annotate(List[List[int]], [])
        for _2 in range(torch.len(tokens_idx0)):
        row = tokens_idx0[_2]
        _3 = torch.append(tokens_idx2, torch.add([self.bos_idx], row))
        tokens_idx1 = tokens_idx2
    else:
        tokens_idx1 = tokens_idx0
    if self.add_eos:
        tokens_idx4 = annotate(List[List[int]], [])
        for _4 in range(torch.len(tokens_idx1)):
        row0 = tokens_idx1[_4]
        _5 = torch.append(tokens_idx4, torch.add(row0, [self.eos_idx]))
        tokens_idx3 = tokens_idx4
    else:
        tokens_idx3 = tokens_idx1
    return tokens_idx3
    >>>>>>>>>>>>>>>>>>>>>>>>>>OUTPUT of JIT code<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    """
    def __init__(
            self,
            vocab_path: Optional[str] = None,
            vocab_list: Optional[List[str]] = None,
            special_token_replacements=SPECIAL_TOKEN_REPLACEMENT,
            add_bos: bool = False,
            add_eos: bool = False,
            max_seq_len: int = 2 ** 30,
    ):
        super().__init__()
        assert vocab_path or vocab_list, "vocab_path or vocab_list is required"
        assert not (
                vocab_path and vocab_list
        ), "vocab_path and vocab_list are mutual exclusive"

        if vocab_list:
            self.vocab = build_vocab_from_iterator([vocab_list],unk_token=UNK)
        else:
            path_manager = PathManager()
            with path_manager.open(vocab_path, "r",encoding='utf-8') as f:
                self.vocab = load_vocab_from_file(f,unk_token=UNK)
 
        self.vocab.append_token(BOS)
        self.vocab.append_token(EOS)
        self.vocab.append_token(PAD)
        self.vocab.append_token(MASK)

        self.bos_idx = self.vocab([BOS])[0]
        self.eos_idx = self.vocab([EOS])[0]

        # TODO T77728853 We need to combine truncate with BOS/EOS as they impact each other
        # Need to find a nicer way to do this, as this can't be chained.
        self.add_bos = add_bos
        self.add_eos = add_eos
        # Make room for bos and eos from max_seq_len if true
        self.truncate_transform = TruncateTransform(max_seq_len - add_bos - add_eos)

        self.vocab = self.vocab.to_ivalue() #Remove to_ivalue() PR: https://github.com/pytorch/text/pull/1080

    @torch.jit.export
    def __len__(self) -> int:
        r"""
        Returns:
            length (int): the length of the vocab
        """
        return len(self.vocab)

    def forward(self, tokens: List[List[str]]) -> List[List[int]]:
        tokens_idx = [self.vocab(token) for token in tokens]
        tokens_idx = self.truncate_transform(tokens_idx)
        if self.add_bos:
            tokens_idx = [[self.bos_idx] + row for row in tokens_idx]
        if self.add_eos:
            tokens_idx = [row + [self.eos_idx] for row in tokens_idx]
        return tokens_idx