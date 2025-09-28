import argparse
import logging
import os
import pathlib
import random
import time

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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


class TokenWindowDataset(Dataset):
    def __init__(self, path: pathlib.Path, block_size: int, dtype: np.dtype = np.uint16):
        self.ids = np.memmap(path, dtype=dtype, mode="r")
        self.block_size = block_size
        self.n = self.ids.shape[0]
        self.num_windows = max(0, (self.n - (block_size + 1)) + 1)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        x = torch.from_numpy(self.ids[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.ids[idx + 1:idx + 1 + self.block_size].astype(np.int64))
        return x, y


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


@torch.no_grad()
def evaluate_perplexity(model: nn.Module, val_loader: DataLoader, device: str = "cuda"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    nll_loss = nn.NLLLoss(ignore_index=-100, reduction="sum")

    for x, y in tqdm(val_loader, desc="evaluating", unit="batch"):
        x = x.to(device)
        y = y.to(device)

        # model must return log-probs: (B, T, V)
        log_probs = model(x)

        loss = nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            y.view(-1)
        )

        total_loss += loss.item()
        total_tokens += y.sum().item()

    avg_nll = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_nll))
    return ppl.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--token-directory', type=pathlib.Path, default=pathlib.Path('.'), help='the location of the tokens')
    parser.add_argument('--checkpoint-directory', type=pathlib.Path, default=pathlib.Path('./checkpoints'), help='the location to store model checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='the number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='the batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='the initial learning rate')
    parser.add_argument('--context', type=int, default=512, help='the context size of the transformer')
    parser.add_argument('--head', type=int, default=4, help='the head size of the transformer')
    parser.add_argument('--layers', type=int, default=6, help='the number of layers of the transformer')
    parser.add_argument('--model-dimension', type=int, default=256, help='the dimension of the model')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of the transformer')
    parser.add_argument('--seed', type=int, default=42, help='the random seed')
    parser.add_argument('--vocab-size', type=int, default=1000, help='the size of the vocab')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_directory, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = 'cpu' if not use_cuda else 'cuda'
    logger.info("training classifier")
    logger.info(f"training on: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f'Loading tokens from {str(args.token_directory)}')
    train_ds = TokenWindowDataset(args.token_directory / 'train.bin', args.context)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=False)
    val_ds = TokenWindowDataset(args.token_directory / 'valid.bin', args.context)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=False)

    logger.info("setting up network")
    model = TransformerModel(
        args.vocab_size, args.context,
        args.model_dimension, args.model_dimension * 4,
        args.head, args.layers,
        args.dropout
    ).to(device)

    logger.info("setting up optimizer")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = LinearLR(optimizer, 0.01, 1.0, total_iters=1000)
    loss_fcn = nn.NLLLoss()  # ignore_index=padding_idx)

    logger.info("starting training loop")
    times = []
    for epoch in range(args.epochs):
        time_start = time.time()
        loss_this_epoch = 0.0
        random.seed(args.seed + epoch)

        bi = 0
        pbar = tqdm(total=len(train_loader), desc=f"training epoch {epoch + 1}", unit="batch")
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            probs = model(xb)
            loss = loss_fcn(
                probs.view(-1, probs.size(-1)),  # (B*T, V)
                yb.view(-1)  # (B*T,)
            )
            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if bi % 1000 == 0:
                pbar.close()
                model.eval()
                perplexity, v_loss = evaluate_perplexity(model, val_loader, device)
                logger.info(f'epoch {epoch + 1}, batch {bi}, loss {loss.item():.4f}, perplexity {perplexity:.4f}')
                logger.info(f'saving model to {args.checkpoint_directory}')
                model_file_name = args.checkpoint_directory / f'model-{epoch}-partial.pt'
                torch.save(model, model_file_name)
                model.train()
                pbar = tqdm(total=len(train_loader), desc=f"training epoch {epoch + 1}", unit="batch")
                pbar.update(bi)

            loss_this_epoch += loss.item()
            scheduler.step()
            bi += 1
            pbar.update(1)

        model.eval()

        perplexity = evaluate_perplexity(model, val_loader, device)

        time_stop = time.time()
        runtime = time_stop - time_start
        times.append(runtime)
        logger.info(
            f"epoch {epoch} ({runtime:.3f} sec): lr: {scheduler.get_last_lr()[0]}, train loss: {loss_this_epoch}, dev perplexity: {perplexity:.3f}")
        if perplexity <= 7.0:  # or v_loss < 30.0:
            logger.info('required perplexity met, stopping training')
            break

        logger.info(f'saving model to {args.checkpoint_directory}')
        model_file_name = args.checkpoint_directory / f'model-{epoch}.pt'
        torch.save(model, model_file_name)

        model.train()