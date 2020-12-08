import os
from collections import defaultdict

import torch
import datasets
from pytorch_lightning import LightningDataModule
from stl_text.ops.tokenizers import WhitespaceTokenizer
from stl_text.ops.transforms import LabelTransform
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence


class DocClassificationDataModule(LightningDataModule):
    def __init__(self, data_path: str = 'glue_sst2_tiny', batch_size: int = 32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

        self.text_transform = None
        self.label_transform = None
        self.datasets = {}

    def setup(self, stage):
        # torchtext's spm_tokenizer and vocab are not pickleable and will fail dataset.map()
        # tokenizer = spm_tokenizer(download_from_url(PRETRAINED_SP_MODEL["text_unigram_25000"]))
        # vocab = build_fairseq_vocab("vocab_tiny.txt")

        # sequential_transforms is not torchscriptable
        # self.text_transform = sequential_transforms(tokenizer, vocab)

        # use dummy text_transform and label_transform to get rid of pickling and torchscript issues
        self.text_transform = WhitespaceTokenizer()
        self.label_transform = LabelTransform(["0", "1"])

        for split in ("train", "valid", "test"):
            self.datasets[split] = datasets.Dataset.load_from_disk(os.path.join(self.data_path, split))  # raw dataset
            self.datasets[split] = self.datasets[split].map(function=lambda x: {'label_id': self.label_transform(x)},
                                                            input_columns='label')
            self.datasets[split] = self.datasets[split].map(function=lambda x: {'token_ids': self.text_transform(x)},
                                                            input_columns='text')
            self.datasets[split].set_format(type='torch', columns=['token_ids', 'label_id'])

    def forward(self, text):
        return self.text_transform(text)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.datasets["train"], shuffle=True, batch_size=self.batch_size,
                                           num_workers=1,
                                           collate_fn=self.collate)

    def valid_dataloader(self):
        return torch.utils.data.DataLoader(self.self.datasets["valid"], shuffle=True, batch_size=self.batch_size,
                                           num_workers=1,
                                           collate_fn=self.collate)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.datasets["test"], shuffle=False, batch_size=self.batch_size,
                                           num_workers=1,
                                           collate_fn=self.collate)

    def collate(self, batch, pad_columns=("token_ids",)):
        columnar_data = defaultdict(list)
        for row in batch:
            for column, value in row.items():
                columnar_data[column].append(value)

        padded = {}
        for column, v in columnar_data.items():
            if pad_columns and column in pad_columns:
                padded[column] = pad_sequence(v, batch_first=True)
            else:
                padded[column] = torch.tensor(v, dtype=torch.long)
        return padded
