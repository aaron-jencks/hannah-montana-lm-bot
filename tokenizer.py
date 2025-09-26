import argparse
import logging
import os
import pathlib

from tokenizers.implementations import ByteLevelBPETokenizer


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
            '<br>'
        ]
    )

    logger.info('saving tokenizer...')
    os.makedirs(args.output_directory, exist_ok=True)

    # output_fname = (args.out / args.name).with_suffix('.tok')
    logger.info(f'saving tokenizer to {args.output_directory} with prefix {args.name}')

    # Save the files
    tokenizer.save_model(str(args.output_directory), args.name)

    logger.info('done')