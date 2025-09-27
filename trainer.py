import argparse
import logging
import pathlib
from typing import List

import numpy as np
import torch
from torch import nn
from transformers import GPT2TokenizerFast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tokenizer(directory: pathlib.Path, name: str) -> GPT2TokenizerFast:
    logger.info('reloading tokenizer...')

    vocab_fname = directory / "{}-vocab.json".format(name)
    merges_fname = directory / "{}-merges.txt".format(name)

    logger.info(f'reading data from {str(vocab_fname)} and {str(merges_fname)}')

    tokenizer = GPT2TokenizerFast(
        str(vocab_fname), str(merges_fname),
        add_prefix_space=True,
        bos_token='<document>',
        eos_token='</document>',
        unk_token='<unk>',
    )
    tokenizer.add_special_tokens({
        'pad_token': '<pad>',
        'additional_special_tokens': [
            '<title>',
            '</title>',
            '<story>',
            '</story>',
            '<br>',
        ],
    })

    return tokenizer


# https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
            self,
            vocab_size: int, seq_length: int,
            d_model: int, d_hidden: int, n_heads: int, layers: int,
            dropout: float
        ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_hidden, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.input_mask = nn.Buffer(torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool())
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
        :param x: shape ``[batch_size, seq_len]``
        :return:
        """
        x = self.embed(x) # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x) # [batch_size, seq_len, d_model]
        x = self.encoder(x, mask=self.input_mask) # [batch_size, seq_len, d_model]
        x = self.linear(x) # [batch_size, seq_len, vocab_size]
        x = self.softmax(x)
        return x


class ModelWrapper:
    def __init__(self, model: TransformerModel, tokenizer: GPT2TokenizerFast):
        self.model = model
        self.tokenizer = tokenizer

    def get_next_character_probs(self, context: str) -> np.ndarray:
        idxs = self.tokenizer(context, return_tensors='pt')
        input_tensor = idxs['input_ids']
        out = self.model(input_tensor)
        probs = out.squeeze(0)[-1].cpu().detach().numpy()
        return probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer-directory', type=pathlib.Path, default=pathlib.Path('.'), help='the location of the tokenizer files')
    parser.add_argument('--tokenizer-name', type=str, default='bpe-bytelevel', help='the name of the tokenizer')
    parser.add_argument('--token-directory', type=pathlib.Path, default=pathlib.Path('.'), help='the location of the tokens')
    parser.add_argument('--checkpoint-directory', type=pathlib.Path, default=pathlib.Path('.'), help='the location to store model checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='the number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64, help='the batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='the initial learning rate')
    parser.add_argument('--context', type=int, default=1024, help='the context size of the transformer')
    parser.add_argument('--head', type=int, default=12, help='the head size of the transformer')
    parser.add_argument('--layers', type=int, default=12, help='the number of layers of the transformer')
    parser.add_argument('--model-dimension', type=int, default=768, help='the dimension of the model')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of the transformer')
    parser.add_argument('--seed', type=int, default=42, help='the random seed')
    parser.add_argument('--vocab-size', type=int, default=1000, help='the size of the vocab')
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.token_directory, args.tokenizer_name)