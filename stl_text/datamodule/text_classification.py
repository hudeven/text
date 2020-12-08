import io
import torch
import datasets as ds
from torchtext.experimental.vocab import build_vocab_from_iterator
from torchtext.experimental.transforms import basic_english_normalize
from torchtext.utils import download_from_url, unicode_csv_reader
from pytorch_lightning import LightningDataModule


def create_data_from_csv(data_path):
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            yield (int(row[0]), ' '.join(row[1:]))


def convert_to_arrow(file_path, raw_data):
    """ Write labels and texts into HF dataset"""
    labels, texts = zip(*raw_data)
    return ds.Dataset.from_dict(
        {
            "labels": labels,
            "texts": texts
        }).save_to_disk(file_path)


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
    def __init__(self, train_valid_split=0.9):
        super().__init__()
        self.train_valid_split = train_valid_split
        self.base_url = 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/'
        self.train_filepath = download_from_url(self.base_url + 'train.csv')
        self.test_filepath = download_from_url(self.base_url + 'test.csv')
        raw_train_data = list(create_data_from_csv(self.train_filepath))
        raw_test_data = list(create_data_from_csv(self.test_filepath))
        train_ds = convert_to_arrow('train_arrow', raw_train_data)
        test_ds = convert_to_arrow('test_arrow', raw_test_data)
        self.tokenizer = basic_english_normalize().to_ivalue()
        train_ds = ds.Dataset.load_from_disk('train_arrow')
        self.vocab = build_vocab_from_iterator(iter(self.tokenizer(line)
                                               for line in train_ds['texts'])).to_ivalue()

    def setup(self, stage):
        # Load and split the raw train dataset into train and valid set
        train_dataset = ds.Dataset.load_from_disk('train_arrow')
        dict_train_valid = train_dataset.train_test_split(test_size=1-self.train_valid_split,
                                                          train_size=self.train_valid_split,
                                                          shuffle=True)
        self.train = dict_train_valid['train']  # raw dataset
        self.valid = dict_train_valid['test']  # raw dataset
        self.test = ds.Dataset.load_from_disk('test_arrow')  # raw dataset

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
