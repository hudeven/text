import torch
import torchtext
import datasets as ds
from torchtext.experimental.transforms import basic_english_normalize
from pytorch_lightning import LightningDataModule
from torchtext.experimental.vocab import load_vocab_from_file


def process_raw_data(arrow_ds, tokenizer, vocab):
    processed_arrow_ds = arrow_ds.map(function=lambda x: {'ans_pos': len(tokenizer(x))}, input_columns='ans_pos')
    processed_arrow_ds = processed_arrow_ds.map(function=lambda x: {'context': vocab(tokenizer(x))}, input_columns='context')
    processed_arrow_ds = processed_arrow_ds.map(function=lambda x: {'question': vocab(tokenizer(x))}, input_columns='question')
    processed_arrow_ds = processed_arrow_ds.map(function=lambda x: {'answers': vocab(tokenizer(x))}, input_columns='answers')
    return processed_arrow_ds


def generate_batch(batch, cls_id, sep_id, pad_id):
    seq_list, ans_pos_list, tok_type = [], [], []
    for item in batch:
        _context, _question = torch.tensor(item['context']), torch.tensor(item['question'])
        qa_item = torch.cat((torch.tensor(cls_id), _question, torch.tensor(sep_id),
                             _context, torch.tensor(sep_id)))
        seq_list.append(qa_item)
        target_start_pos = item['ans_pos'] + _question.size(0) + 2
        target_end_pos = target_start_pos + len(item['answers'])
        ans_pos_list.append(torch.tensor([target_start_pos, target_end_pos], dtype=torch.long))
        tok_type.append(torch.cat((torch.zeros((_question.size(0) + 2)),
                                   torch.ones((_context.size(0) + 1)))))
    ans_pos_list = torch.stack(ans_pos_list)
    target_start_pos, target_end_pos = ans_pos_list.split(1, dim=-1)
    target_start_pos = target_start_pos.squeeze(-1)
    target_end_pos = target_end_pos.squeeze(-1)
    seq_list = torch.nn.utils.rnn.pad_sequence(seq_list, batch_first=True, padding_value=float(pad_id[0]))
    seq_list = seq_list.long().t().contiguous()
    tok_type = torch.nn.utils.rnn.pad_sequence(tok_type, batch_first=True, padding_value=1.0)
    tok_type = tok_type.long().t().contiguous()
    return seq_list, target_start_pos, target_end_pos, tok_type


class QuestionAnswerDataModule(LightningDataModule):
    def __init__(self, train_arrow_path='train_arrow',
                 dev_arrow_path='dev_arrow', vocab_filepath=None, batch_size=8):
        super().__init__()
        self.train_arrow_path = train_arrow_path
        self.dev_arrow_path = dev_arrow_path
        self.tokenizer = basic_english_normalize().to_ivalue()
        with open(vocab_filepath, 'r') as f:
            self.vocab = load_vocab_from_file(f).to_ivalue()
        self.cls_id = self.vocab(['<cls>'])
        self.sep_id = self.vocab(['<sep>'])
        self.pad_id = self.vocab(['<pad>'])
        self.bsz = batch_size

    def setup(self, stage):
        self.train = ds.Dataset.load_from_disk(self.train_arrow_path)  # raw dataset
        self.dev = ds.Dataset.load_from_disk(self.dev_arrow_path)  # raw dataset

    def train_dataloader(self):
        self.train = process_raw_data(self.train, self.tokenizer, self.vocab)
        return torch.utils.data.DataLoader(self.train, shuffle=True,
                                           batch_size=self.bsz, num_workers=1,
                                           collate_fn=lambda x: generate_batch(x, self.cls_id, self.sep_id, self.pad_id))

    def val_dataloader(self):
        self.dev = process_raw_data(self.dev, self.tokenizer, self.vocab)
        return torch.utils.data.DataLoader(self.dev, batch_size=self.bsz, num_workers=1,
                                           collate_fn=lambda x: generate_batch(x, self.cls_id, self.sep_id, self.pad_id))
