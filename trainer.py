import argparse
import datetime
import logging
import os
import pathlib
import random
import time
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import GPT2TokenizerFast
import wandb

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


class SampledTokenWindowDataset(TokenWindowDataset):
    def __init__(self, path: pathlib.Path, block_size: int, dtype: np.dtype = np.uint16, samples: int = 2048):
        super().__init__(path, block_size, dtype)
        self.samples = samples
        if self.samples < self.num_windows:
            logger.warning(f'number of samples exceeds number of windows ({self.samples} > {self.num_windows})')
        self.sample_ids = np.asarray(random.sample(list(range(self.num_windows)), samples))

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        window_idxs = self.sample_ids[idx]
        return super().__getitem__(window_idxs)


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
        # self.input_mask = nn.Buffer(torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool())
        # self.register_buffer('input_mask', self.input_mask, persistent=False)
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor, key_padding_mask = None) -> torch.Tensor:
        """
        Arguments:
        :param x: shape ``[batch_size, seq_len]``
        :param key_padding_mask: shape ``[batch_size, seq_len]``
        :return:
        """
        S = x.size(1)
        x = self.embed(x) # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x) # [batch_size, seq_len, d_model]
        attn_mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        if key_padding_mask is not None:
            x = self.encoder(x, mask=attn_mask, is_causal=True, src_key_padding_mask=key_padding_mask)
        else:
            x = self.encoder(x, mask=attn_mask, is_causal=True) # [batch_size, seq_len, d_model]
        x = self.linear(x) # [batch_size, seq_len, vocab_size]
        x = self.softmax(x)
        return x


@torch.no_grad()
def evaluate_perplexity(model: nn.Module, val_loader: DataLoader, device: str = "cuda", use_bar: bool = True) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    pbar = None
    if use_bar:
        pbar = tqdm(total=len(val_loader), desc="evaluating", unit="batch")
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        # model must return log-probs: (B, T, V)
        log_probs = model(x)

        loss = nn.functional.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            y.view(-1),
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += y.numel()
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()

    avg_nll = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_nll))
    return ppl.item(), avg_nll


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, default='debug', help="run name")
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
    parser.add_argument('--resume', type=pathlib.Path, default=None, help='the location of the pretrained model to resume')
    parser.add_argument('wandb_team', type=str, help='the name of the wandb team to push to')
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
    mini_val_ds = SampledTokenWindowDataset(args.token_directory / 'valid.bin', args.context)
    mini_val_loader = DataLoader(mini_val_ds, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=False)

    logger.info("setting up network")
    if args.resume:
        logger.info(f"loading checkpoint {args.resume}")
        model = torch.load(args.resume, weights_only=False).to(device)
    else:
        model = TransformerModel(
            args.vocab_size, args.context,
            args.model_dimension, args.model_dimension * 4,
            args.head, args.layers,
            args.dropout
        ).to(device)

    logger.info("setting up optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.1)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=400  # 400 warmup steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=int(len(train_loader) * 0.9), T_mult=1, eta_min=args.lr * 0.1  # endless cycles
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[400]
    )
    loss_fcn = nn.NLLLoss()  # ignore_index=padding_idx)

    logger.info('setting up training optimizations...')
    eval_steps = len(train_loader) // 100
    logger.info(f'will eval every {eval_steps} steps')
    scaler = torch.amp.GradScaler(device)

    logger.info('setting up wandb...')
    name = f"{datetime.datetime.now().strftime('%m-%d-%Y T %H:%M:%S')} {args.run_name}"
    logger.info(f'run name: {name}')
    # os.environ["WANDB_CONSOLE"] = "off"
    run = wandb.init(
        project="hannah-montana-pretrain",  # your project bucket
        entity=args.wandb_team,  # your personal username or the team name
        name=name,  # optional
        config={
            "lr": args.lr,
            "ctx": args.context,
            "batch": args.batch_size,
            "heads": args.head,
            "d_model": args.model_dimension,
            "layers": args.layers,
            "dropout": args.dropout,
            "seed": args.seed,
            "vocab": args.vocab_size,
        },
        # settings=wandb.Settings(console="off")
    )
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("tiny_val/*", step_metric="global_step")
    wandb.define_metric("tiny_val/ppl", summary="min")
    wandb.define_metric("val/*", step_metric="global_step")
    wandb.define_metric("val/ppl", summary="min")

    logger.info("starting training loop")
    times = []
    global_step = 0
    for epoch in range(args.epochs):
        time_start = time.time()
        loss_this_epoch = 0.0
        random.seed(args.seed + epoch)

        bi = 0
        pbar = tqdm(total=len(train_loader), desc=f"training epoch {epoch + 1}", unit="batch")
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device):
                probs = model(xb)
                loss = loss_fcn(
                    probs.view(-1, probs.size(-1)),  # (B*T, V)
                    yb.view(-1)  # (B*T,)
                )
            wandb.log({"train/loss": loss.item(), "epoch": epoch, "global_step": global_step, "learning_rate": scheduler.get_last_lr()[0]})
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if bi % eval_steps == 0:
                pbar.clear()
                model.eval()
                perplexity, vloss = evaluate_perplexity(model, mini_val_loader, device, False)
                wandb.log({"tiny_val/ppl": perplexity, "tiny_val/loss": vloss, "epoch": epoch, "global_step": global_step})
                logger.info(f'epoch {epoch + 1}, batch {bi}, loss {loss.item():.4f}, perplexity {perplexity:.4f}')
                logger.info(f'saving model to {args.checkpoint_directory}')
                model_file_name = args.checkpoint_directory / f'model-{epoch}-partial.pt'
                torch.save(model, model_file_name)
                model.train()
                pbar.refresh()

            loss_this_epoch += loss.item()
            scheduler.step()
            bi += 1
            pbar.update(1)
            global_step += 1

        pbar.close()
        model.eval()

        perplexity, vloss = evaluate_perplexity(model, val_loader, device)

        time_stop = time.time()
        runtime = time_stop - time_start
        times.append(runtime)
        wandb.log({"val/ppl": perplexity, "val/loss": vloss, "epoch": epoch, "global_step": global_step})
        logger.info(f"epoch {epoch} ({runtime:.3f} sec): lr: {scheduler.get_last_lr()[0]}, train loss: {loss_this_epoch}, dev perplexity: {perplexity:.3f}")

        logger.info(f'saving model to {args.checkpoint_directory}')
        model_file_name = args.checkpoint_directory / f'model-{epoch}.pt'
        torch.save(model, model_file_name)

        model.train()

    run.finish()