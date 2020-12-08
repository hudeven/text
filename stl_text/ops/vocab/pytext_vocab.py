#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional

import torch.nn as nn
from pytext.data.bert_tensorizer import build_fairseq_vocab
from pytext.data.utils import BOS, EOS, MASK, PAD, UNK
from pytext.torchscript.vocab import ScriptVocabulary
from pytext.utils.file_io import PathManager

SPECIAL_TOKEN_REPLACEMENT = {
    "[UNK]": UNK,
    "[PAD]": PAD,
    "[CLS]": BOS,
    "[MASK]": MASK,
    "[SEP]": EOS,
}


class VocabTransform(nn.Module):
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
            self.vocab = ScriptVocabulary(vocab_list)
        else:
            with PathManager.open(vocab_path) as f:
                vocab = build_fairseq_vocab(
                    f, special_token_replacements=special_token_replacements
                )
                self.vocab = ScriptVocabulary(
                    list(vocab),
                    pad_idx=vocab.get_pad_index(-1),
                    bos_idx=vocab.get_bos_index(-1),
                    eos_idx=vocab.get_eos_index(-1),
                    unk_idx=vocab.get_unk_index(-1),
                    unk_token=vocab.unk_token,
                )
        # TODO T77728853 We need to combine truncate with BOS/EOS as they impact each other
        # Need to find a nicer way to do this, as this can't be chained.
        self.add_bos = add_bos
        self.add_eos = add_eos
        # Make room for bos and eos from max_seq_len if true
        self.truncate_transform = TruncateTransform(max_seq_len - add_bos - add_eos)

    def forward(self, tokens: List[List[str]]) -> List[List[int]]:
        tokens_idx = self.vocab.lookup_indices_2d(tokens)
        tokens_idx = self.truncate_transform(tokens_idx)
        if self.add_bos:
            tokens_idx = [[self.vocab.bos_idx] + row for row in tokens_idx]
        if self.add_eos:
            tokens_idx = [row + [self.vocab.eos_idx] for row in tokens_idx]
        return tokens_idx


class TruncateTransform(nn.Module):
    def __init__(self, max_seq_len: int):
        super().__init__()
        assert max_seq_len > 0
        self.max_seq_len: int = max_seq_len

    def forward(self, token_ids: List[List[int]]) -> List[List[int]]:
        return [token_id[: self.max_seq_len] for token_id in token_ids]
