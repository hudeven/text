import torch
import torchtext
import datasets as ds
from torchtext.experimental.vocab import build_vocab_from_iterator
from torchtext.experimental.transforms import basic_english_normalize
from pytorch_lightning import LightningDataModule


def process_raw_data(arrow_ds, tokenizer, vocab):
    processed_arrow_ds = arrow_ds.map(function=lambda x: {'labels': int(x) - 1}, input_columns='labels')
    processed_arrow_ds = processed_arrow_ds.map(function=lambda x: {'texts': vocab(tokenizer(x))}, input_columns='texts')
    return processed_arrow_ds


def generate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for item in batch:
        label_list.append(item['labels'])
        processed_text = torch.tensor(item['texts'], dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets


class TextClassificationDataModule(LightningDataModule):
    def __init__(self, train_arrow_path='train_arrow',
                 test_arrow_path='test_arrow',
                 train_valid_split=0.9):
        super().__init__()
        self.train_arrow_path = train_arrow_path
        self.test_arrow_path = test_arrow_path
        self.train_valid_split = train_valid_split
        self.tokenizer = basic_english_normalize().to_ivalue()
        train_ds = ds.Dataset.load_from_disk(self.train_arrow_path)
        self.vocab = build_vocab_from_iterator(iter(self.tokenizer(line)
                                               for line in train_ds['texts'])).to_ivalue()

    def setup(self, stage):
        # Load and split the raw train dataset into train and valid set
        train_dataset = ds.Dataset.load_from_disk(self.train_arrow_path)
        dict_train_valid = train_dataset.train_test_split(test_size=1-self.train_valid_split,
                                                          train_size=self.train_valid_split,
                                                          shuffle=True)
        self.train = dict_train_valid['train']  # raw dataset
        self.valid = dict_train_valid['test']  # raw dataset
        self.test = ds.Dataset.load_from_disk(self.test_arrow_path)  # raw dataset

    def train_dataloader(self):
        # Process the raw dataset
        self.train = process_raw_data(self.train, self.tokenizer, self.vocab)
        return torch.utils.data.DataLoader(self.train, shuffle=True,
                                           batch_size=16, num_workers=1,
                                           collate_fn=generate_batch)

    def val_dataloader(self):
        # Process the raw dataset
        self.valid = process_raw_data(self.valid, self.tokenizer, self.vocab)
        return torch.utils.data.DataLoader(self.valid, batch_size=16, num_workers=1,
                                           collate_fn=generate_batch)

    def test_dataloader(self):
        # Process the raw dataset
        self.test = process_raw_data(self.test, self.tokenizer, self.vocab)
        return torch.utils.data.DataLoader(self.test, batch_size=16, num_workers=1,
                                           collate_fn=generate_batch)
