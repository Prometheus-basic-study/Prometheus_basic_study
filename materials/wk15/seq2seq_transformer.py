import numpy as np
import pandas as pd

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import re, string
from unicodedata import normalize

from typing import Optional, List, Tuple, Dict, Iterable, Callable

import os
DATA_PATH = 'data'
if not os.path.exists('data'):
    DATA_PATH = os.path.join('materials', 'wk15', DATA_PATH)

def clean_lines(lines):
    if isinstance(lines, list):
        return [clean_lines(line) for line in lines]

    is_question = lines.endswith('?')
    remove_punctuation = str.maketrans('', '', string.punctuation)
    lines = normalize('NFD', lines).encode('ascii', 'ignore')
    lines = lines.decode('UTF-8')
    lines = lines.lower()
    lines = lines.translate(remove_punctuation)
    lines = re.sub(rf'[^{re.escape(string.printable)}]', '', lines)

    lines = [word for word in lines.split() if word.isalpha()]
    if is_question:
        lines.append('?')
    return ' '.join(lines)


from torchtext.data.utils import get_tokenizer

train_df, valid_df, test_df = pd.read_pickle(os.path.join(DATA_PATH, 'small_train_valid_test.pkl'))
en_vocab, fr_vocab = pd.read_pickle(os.path.join(DATA_PATH, 'small_vocab.pkl'))
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

SRC_LANG = 'fr'
TGT_LANG = 'en'

vocab_transform = {
    # List[str] -> List[int]
    'fr': fr_vocab,
    'en': en_vocab
}

tokenizer_transform = {
    # str -> List[str]
    'fr': fr_tokenizer,
    'en': en_tokenizer
}

from torch.nn.utils.rnn import pad_sequence

BOS_IDX = vocab_transform[TGT_LANG]['<bos>']
EOS_IDX = vocab_transform[TGT_LANG]['<eos>']
PAD_IDX = vocab_transform[TGT_LANG]['<pad>']

def sequential_transforms(*transforms):
    # Compose several transforms to be applied sequentially.
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def append_special(token_ids: List[int]):
    return th.cat([
        th.tensor([BOS_IDX], dtype=th.long),
        th.tensor(token_ids),
        th.tensor([EOS_IDX], dtype=th.long)
    ])

text_transform = {lang: sequential_transforms(tokenizer_transform[lang],
                                               vocab_transform[lang],
                                               append_special)
                    for lang in [SRC_LANG, TGT_LANG]}

def collate_fn(batch):
    """
    Collate function defines how to process a batch of data into a batch of tensors.
    """
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANG](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANG](tgt_sample.rstrip("\n")))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

from torch.utils.data import DataLoader

BATCH_SIZE = 32
train_iter = DataLoader(list(zip(train_df[SRC_LANG], train_df[TGT_LANG])),
                        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

valid_iter = DataLoader(list(zip(valid_df[SRC_LANG], valid_df[TGT_LANG])),
                        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

test_iter = DataLoader(list(zip(test_df[SRC_LANG], test_df[TGT_LANG])),
                          batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 batch_first: bool = False):

        super(PositionalEncoding, self).__init__()
        self.batch_first = batch_first
        density = th.exp(-th.arange(0, emb_dim, 2) * np.log(10000) / emb_dim)
        pos = th.arange(0, max_len).unsqueeze(1)
        pos_embedding = th.zeros((max_len, emb_dim))
        pos_embedding[:, 0::2] = th.sin(pos * density)
        pos_embedding[:, 1::2] = th.cos(pos * density)
        pos_embedding = pos_embedding.unsqueeze(-2)  # [max_len, 1, emb_dim]

        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        # x = [seq_len, batch_size, emb_dim] or [batch_size, seq_len, emb_dim]
        if self.batch_first:
            return self.dropout(x + self.pos_embedding[:x.size(1), :].permute(1, 0, 2))
        else:
            return self.dropout(x + self.pos_embedding[:x.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super(TokenEmbedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x):
        # x = [seq_len, batch_size]
        return self.emb(x.long()) * np.sqrt(self.emb_dim)

    def forward(self, x):
        # x = [seq_len, batch_size]
        return self.emb(x.long()) * np.sqrt(self.emb_dim)


class TransformerSeq2Seq(nn.Module):
    def __init__(self,
                 num_enc_layers: int,
                 num_dec_layers: int,
                 emb_dim: int,
                 n_heads: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_ff: int = 512,
                 dropout: float = 0.1,
                 batch_first: bool = False):
        super(TransformerSeq2Seq, self).__init__()
        self.batch_first = batch_first
        self.src_emb = nn.Sequential(TokenEmbedding(src_vocab_size, emb_dim),
                                     PositionalEncoding(emb_dim, dropout=dropout, batch_first=batch_first))
        self.tgt_emb = nn.Sequential(TokenEmbedding(tgt_vocab_size, emb_dim),
                                     PositionalEncoding(emb_dim, dropout=dropout, batch_first=batch_first))
        self.transformer = nn.Transformer(d_model=emb_dim,
                                          nhead=n_heads,
                                          num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers,
                                          dim_feedforward=dim_ff,
                                          dropout=dropout,
                                          batch_first=batch_first)
        self.regressor = nn.Linear(emb_dim, tgt_vocab_size)

    def forward(self, src: th.Tensor, tgt: th.Tensor,
                src_mask: th.Tensor = None, tgt_mask: th.Tensor = None,
                src_padding_mask: th.Tensor = None,
                tgt_padding_mask: th.Tensor = None,
                memory_key_padding_mask: th.Tensor = None) -> th.Tensor:
        """

        :param src:         [seq_len, bs] or [bs, seq_len]
        :param tgt:         [seq_len, bs] or [bs, seq_len]
        :param src_mask:    [seq_len, bs, seq_len] or [bs, seq_len, seq_len]
        :param tgt_mask:    [seq_len, bs, seq_len] or [bs, seq_len, seq_len]
        :param src_padding_mask:    [seq_len, bs] or [bs, seq_len]
        :param tgt_padding_mask:    [seq_len, bs] or [bs, seq_len]
        :param memory_key_padding_mask: [seq_len, bs] or [bs, seq_len]
        :return:    [seq_len, bs, vocab_size] or [bs, seq_len, vocab_size]
        """
        # src = [seq_len, batch_size] or [batch_size, seq_len]
        # tgt = [seq_len, batch_size] or [batch_size, seq_len]
        src = self.src_emb(src)
        tgt = self.tgt_emb(tgt)
        # [seq_len, bs, emb_dim] or [bs, seq_len, emb_dim]
        output = self.transformer(src, tgt,
                                  src_mask=src_mask, tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)

        output = self.regressor(output)
        return output

    def encode(self, src: th.Tensor, src_mask: th.Tensor):
        src = self.src_emb(src)
        return self.transformer.encoder(src, mask=src_mask)

    def decode(self, tgt: th.Tensor, memory: th.Tensor, tgt_mask: th.Tensor = None, memory_mask: th.Tensor=None):
        tgt = self.tgt_emb(tgt)
        return self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)


def look_ahead_mask(seq_len: int, device: th.device = th.device('cpu')):
    # look ahead mask for decoder 1'st layer
    mask = th.triu(th.ones((seq_len, seq_len), device=device)).transpose(0, 1)
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src: th.Tensor, tgt: th.Tensor, pad_idx: int = PAD_IDX, batch_first: bool = False):
    """
    src_mask: all True mask, [seq_len, seq_len]
    tgt_mask: look ahead mask, [seq_len, seq_len]
    src_padding_mask: [seq_len, bs] or [bs, seq_len]
    tgt_padding_mask: [seq_len, bs] or [bs, seq_len]
    """
    src_seq_len = src.size(-1 if batch_first else 0)
    tgt_seq_len = tgt.size(-1 if batch_first else 0)

    tgt_mask = look_ahead_mask(tgt_seq_len, device=tgt.device)
    src_mask = th.ones((src_seq_len, src_seq_len), device=src.device)

    src_padding_mask = (src == pad_idx)
    tgt_padding_mask = (tgt == pad_idx)
    if not batch_first:
        src_padding_mask = src_padding_mask.transpose(0, 1)
        tgt_padding_mask = tgt_padding_mask.transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


print(look_ahead_mask(6))
src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(th.tensor([5,6,7,8,1,1]).view(-1, 1).repeat(1, 2),
                                                             th.tensor([1,1,7,8,9,10]).view(-1, 1).repeat(1, 2),
                                                             batch_first=False)
print(f'src_mask:\n{src_mask}')
print(f'tgt_mask:\n{tgt_mask}')
print(f'src_pad_mask:\n{src_pad_mask}')
print(f'tgt_pad_mask:\n{tgt_pad_mask}')


th.manual_seed(42)

src_vocab_size = len(vocab_transform[SRC_LANG])
tgt_vocab_size = len(vocab_transform[TGT_LANG])
emb_dim = 512
n_heads = 8
num_enc_layers = 3
num_dec_layers = 3
dim_ff = 512
dropout = 0.1

batch_size = 128

model = TransformerSeq2Seq(num_enc_layers=num_enc_layers,
                           num_dec_layers=num_dec_layers,
                           emb_dim=emb_dim,
                           n_heads=n_heads,
                           src_vocab_size=src_vocab_size,
                           tgt_vocab_size=tgt_vocab_size,
                           dim_ff=dim_ff,
                           dropout=dropout)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
# to see the def of Adam, see https://velog.io/@viriditass/%EB%82%B4%EA%B0%80-%EB%B3%B4%EB%A0%A4%EA%B3%A0-%EB%A7%8C%EB%93%A0-Optimizier-%EC%A0%95%EB%A6%AC
optimizer = th.optim.Adam(model.parameters(), lr=0.0001,
                          betas=(0.9, 0.98), eps=1e-9)

from tqdm import tqdm
from torch.utils.data import DataLoader

def train(model: TransformerSeq2Seq,
          dataloader: DataLoader,
          optimizer: th.optim.Optimizer,
          criterion: nn.Module,
          device: th.device = th.device('cpu')):

    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader, total=len(dataloader), postfix='Train: ')
    for i, (src, tgt) in enumerate(progress_bar):
        src = src.to(device)
        tgt = tgt.to(device)

        decoder_input = tgt[:-1, :] # exclude the last token
        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, decoder_input)
        # [seq_len, bs, tgt_vocab_size]
        logits = model(src, decoder_input, src_mask, tgt_mask,
                       src_padding_mask=src_pad_mask,
                       tgt_padding_mask=tgt_pad_mask,
                       memory_key_padding_mask=src_pad_mask)

        optimizer.zero_grad()
        tgt_out = tgt[1:, :].view(-1) # exclude the first token
        output = logits.view(-1, logits.shape[-1])
        loss = criterion(output, tgt_out)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_description(f'loss: {epoch_loss / (i + 1):.3f}')

        if i % 30 == 0:
            th.cuda.empty_cache()

    return epoch_loss / len(dataloader)

def evaluate(model: TransformerSeq2Seq,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: th.device = th.device('cpu')):

    model.eval()
    losses = 0
    progress_bar = tqdm(dataloader, total=len(dataloader), postfix='Eval: ')
    with th.no_grad():
        for i, (src, tgt) in enumerate(progress_bar):
            src = src.to(device)
            tgt = tgt.to(device)

            decoder_input = tgt[:-1, :] # exclude the last token
            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt)
            logits = model(src, decoder_input, src_mask, tgt_mask,
                           src_padding_mask=src_pad_mask,
                           tgt_padding_mask=tgt_pad_mask,
                           memory_key_padding_mask=src_pad_mask)

            tgt_out = tgt[1:, :].view(-1) # exclude the first token
            output = logits.view(-1, logits.shape[-1])
            loss = criterion(output, tgt_out)
            losses += loss.item()
            progress_bar.set_description(f'loss: {losses / (i + 1):.3f}')

    th.cuda.empty_cache()
    return losses / len(dataloader)


def greedy_decode(model: TransformerSeq2Seq,
                  src: th.Tensor,
                  src_mask: th.Tensor,
                  max_len: int,
                  start_symbol: int,
                  device: th.device = th.device('cpu')):

    memory = model.encode(src, src_mask)
    ys = th.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        tgt_mask = look_ahead_mask(ys.size(0), device=device)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        out = model.decode(ys, memory, tgt_mask)
        prob = model.regressor(out[:, -1])
        _, next_word = th.max(prob, dim = 1)
        next_word = next_word.item()
        ys = th.cat([ys, th.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


def translate(model:TransformerSeq2Seq,
              src_sentence: str):
    
    model.eval()
    src = text_transform[SRC_LANG](src_sentence).view(-1, 1).to(device)
    num_tokens = src.shape[0]   # if not batch_first
    src_mask = th.zeros(num_tokens, num_tokens).type_as(src.data)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    out_sentence = " ".join(vocab_transform[TGT_LANG].lookup_tokens(list(tgt_tokens.cpu().numpy())))
    return out_sentence
    

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
model = model.to(device)
# train(model, train_iter, optimizer, criterion, device)
out = translate(model, 'my name is anonymous')