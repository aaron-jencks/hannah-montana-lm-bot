import argparse
import logging
import os
import pathlib
import random
from typing import List

import numpy as np
from tokenizers.implementations import ByteLevelBPETokenizer
from tqdm import tqdm
from transformers import GPT2TokenizerFast


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def flatten_documents(documents) -> List[int]:
    result = []
    for document in documents:
        result.extend(document)
    return result


def save_data(split_data: List[int], split: str, directory: pathlib.Path):
    logger.info(f'saving data for {split}...')
    id_fname = directory / f'{split}.bin'
    with open(id_fname, 'wb+') as fp:
        iddata = np.array(split_data, dtype=np.uint16).tobytes()
        fp.write(iddata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=pathlib.Path, default=pathlib.Path('./corpus.txt'), help='Path to the corpus file')
    parser.add_argument(
        "--output-directory",
        default=pathlib.Path("."),
        type=pathlib.Path,
        help="Path to the output directory, where the files will be saved",
    )
    parser.add_argument(
        "--name",
        default="bpe-bytelevel",
        type=str,
        help="The name of the output vocab files"
    )
    parser.add_argument('--cpus', type=int, default=os.cpu_count(), help='the number of cores to use')
    parser.add_argument('--validation-ratio', type=float, default=0.2, help='the ratio of validation set size')
    parser.add_argument('--token-directory', type=pathlib.Path, default=pathlib.Path("."), help='Path to the output token directory')
    parser.add_argument('--vocab-size', type=int, default=1000, help='the size of the vocabulary to train')
    args = parser.parse_args()

    logger.info('training tokenizer...')
    # Initialize an empty tokenizer
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)

    # And then train
    tokenizer.train(
        str(args.input),
        vocab_size=args.vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=[
            '<title>', '</title>',
            '<story>', '</story>',
            '<br>',
            '<document>', '</document>',
        ]
    )

    logger.info('saving tokenizer...')
    os.makedirs(args.output_directory, exist_ok=True)

    # output_fname = (args.out / args.name).with_suffix('.tok')
    logger.info(f'saving tokenizer to {args.output_directory} with prefix {args.name}')

    # Save the files
    tokenizer.save_model(str(args.output_directory), args.name)

    logger.info('applying tokenizer...')
    logger.info('reloading tokenizer...')

    vocab_fname = args.output_directory / "{}-vocab.json".format(args.name)
    merges_fname = args.output_directory / "{}-merges.txt".format(args.name)

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

    with open(args.input, 'r') as fp:
        lines = fp.read().splitlines(keepends=False)

    first_row_ids = tokenizer([lines[0]])['input_ids'][0]
    logger.info(f'First row encoded:')
    logger.info(f'ids: {first_row_ids}')
    logger.info(f'tokens: {tokenizer.convert_ids_to_tokens(first_row_ids)}')

    token_documents = []
    for line in lines:
        tokens = tokenizer([line])['input_ids'][0]
        token_documents.append(tokens)

    train_size = int(len(token_documents) * (1 - args.validation_ratio))
    random.shuffle(token_documents)
    train_documents = token_documents[:train_size]
    valid_documents = token_documents[train_size:]

    train_tokens = flatten_documents(train_documents)
    valid_tokens = flatten_documents(valid_documents)

    save_data(train_tokens, 'train', args.token_directory)
    save_data(valid_tokens, 'valid', args.token_directory)