import torch
from torch.utils.data import DataLoader
from torchtext.utils import download_from_url
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import IMDB
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from plato.config import Config
from plato.datasources import base
from collections import Counter

import io
import os
import tarfile
import zipfile
import gzip
from pathlib import Path

_PATH = 'aclImdb_v1.tar.gz'
URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

MD5 = '7c2ac02c03563afcf9b574c7e56c153a'
NUM_LINES = {
    'train': 25000,
    'test': 25000,
}

DATASET_NAME = "IMDB"
def extract_archive(from_path, to_path=None):

    if to_path is None:
        to_path = os.path.dirname(from_path)

    if from_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(from_path, 'r') as tar:
            files = []
            for file_ in tar:
                file_path = os.path.join(to_path, file_.name)
                if file_.isfile():
                    files.append(file_path)
            return files

    elif from_path.endswith('.zip'):
        with zipfile.ZipFile(from_path, 'r') as zfile:
            files = []
            for file_ in zfile.namelist():
                file_path = os.path.join(to_path, file_)
                files.append(file_path)
        files = [f for f in files if os.path.isfile(f)]
        return files

    elif from_path.endswith('.gz'):
        filename = from_path[:-3]
        files = [filename]
        return files

    else:
        raise NotImplementedError(
            "We currently only support tar.gz, .tgz, .gz and zip achives.")

def IMDB(root, split):
    def generate_imdb_data(key, extracted_files):
        for fname in extracted_files:
            *_, split, label, file = Path(fname).parts

            if key == split and (label in ['pos', 'neg']):
                with io.open(fname, encoding="utf8") as f:
                    yield label, f.read()

    dataset_tar = download_from_url(URL, root=root,
                                    hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)
    iterator = generate_imdb_data(split, extracted_files)
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split], iterator)

class DataSource(base.DataSource):
    """The CIFAR-10 dataset."""
    def __init__(self):
        super().__init__()
        _path = os.path.join(Config().data.data_path, 'IMDB')
        self.trainset = IMDB(root=_path, split='train')
        self.testset = IMDB(root=_path, split='test')

    def num_train_examples(self):
        return 25000

    def num_test_examples(self):
        return 25000

    def targets(self):
        _path = os.path.join(Config().data.data_path, 'IMDB')
        trainset = to_map_style_dataset(IMDB(root=_path, split='train'))
        target_dict = {'neg': 0, 'pos': 1}
        targets = [target_dict[item[0]] for item in trainset]
        return targets

    def classes(self):
        return ['0 - neg', '1 - pos']

class BatchWrapper:
    def __init__(self, dl):
        self.dl = dl

    def __iter__(self):
        for batch in self.dl:
            yield (batch.text, batch.label)

from torch import nn

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.rnn = nn.LSTM(input_size=20, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        x, _ = self.rnn(embedded)
        return self.fc(x[:, -1, :])

# if __name__ == '__main__':
    # dataset = DataSource()
    # print(dataset.targets())
    # tokenizer = get_tokenizer('basic_english')
    # train_iter = IMDB(root='../../examples/momentum_adp/data/IMDB', split='train')
    # # model = TextClassificationModel(20000, 20, 2)
    # # model(torch.zeros((32, 20, 20000)).long())
    #
    #
    # def yield_tokens(data_iter):
    #     for _, text in data_iter:
    #         yield tokenizer(text)
    #
    # # train_iter = to_map_style_dataset(train_iter)
    # vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    # vocab.set_default_index(vocab["<unk>"])
    # text_pipeline = lambda x: vocab(tokenizer(x))
    # label_pipeline = lambda x: 0 if x == 'neg' else 1
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # def collate_batch(batch):
    #     label_list, text_list, offsets = [], [], [0]
    #     for (_label, _text) in batch:
    #         label_list.append(label_pipeline(_label))
    #         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
    #         text_list.append(processed_text)
    #         offsets.append(processed_text.size(0))
    #     label_list = torch.tensor(label_list, dtype=torch.int64)
    #     offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    #     text_list = torch.cat(text_list)
    #     print(text_list.shape)
    #     return label_list.to(device), text_list.to(device), offsets.to(device)
    #
    # train_iter = IMDB(root='../../examples/momentum_adp/data/IMDB', split='train')
    # vocab_size = len(vocab)
    # print(vocab_size)
    # emsize = 64
    # model = TextClassificationModel(vocab_size, emsize, 2).to(device)
    # dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)
    #
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # for idx, (label, text, offsets) in enumerate(dataloader):
    #     print(text.shape, offsets.shape)
    #     optimizer.zero_grad()
    #     predicted_label = model(text, offsets)
    #     loss = criterion(predicted_label, label)
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    #     optimizer.step()