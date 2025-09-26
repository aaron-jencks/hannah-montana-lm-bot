import argparse
import logging
import os
import pathlib

from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    args = parser.parse_args()

    logger.info('training tokenizer...')
    # Initialize an empty tokenizer
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)

    # And then train
    tokenizer.train(
        str(args.input),
        vocab_size=50000,
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

    print(f'reading data from {str(vocab_fname)} and {str(merges_fname)}')

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
    print(f'First row encoded:')
    print(f'ids: {first_row_ids}')
    print(f'tokens: {tokenizer.convert_ids_to_tokens(first_row_ids)}')