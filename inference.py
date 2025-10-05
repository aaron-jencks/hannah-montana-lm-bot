import argparse
import logging
import pathlib
from typing import List

import torch
from transformers import GPT2TokenizerFast

from trainer import TransformerModel, PositionalEncoding


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelWrapper:
    def __init__(self, model: TransformerModel, tokenizer: GPT2TokenizerFast, context_size: int, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context_size = context_size

    def get_context_window(self, current_story: List[int]) -> List[int]:
        # if len(current_story) < self.context_size:
        #     return [tokenizer.pad_token_id] * (self.context_size - len(current_story)) + current_story
        # elif len(current_story) == self.context_size:
        #     return current_story
        # else:
        #     return current_story[-self.context_size:]
        if len(current_story) > self.context_size:
            return current_story[-self.context_size:]
        return current_story

    def generate_story(self, title: str, prefix: str) -> str:
        context = f'<document> <title> {title} </title> <story> {prefix}'
        encoding = self.tokenizer.encode(context)

        new_context = self.get_context_window(encoding)
        input_tensor = torch.LongTensor([new_context]).to(self.device)
        # padding_mask = (input_tensor == self.tokenizer.pad_token_id)

        print(title)
        print(prefix.replace('<br>', '\n'), end='', flush=True)

        while True:
            with torch.no_grad():
                out = self.model(input_tensor)  # , key_padding_mask=padding_mask)
                probs = torch.exp(out.squeeze(0)[-1])
                next_id = torch.multinomial(probs, num_samples=1).item()

            token_str = tokenizer.decode([next_id], clean_up_tokenization_spaces=False)
            # if token_str in [
            #     '</title>',
            #     '<title>',
            #     '<story>',
            #     '<document>',
            # ]:
            #     continue
            encoding.append(next_id)
            if next_id == self.tokenizer.eos_token_id:
                break
            if token_str == '<br>':
                print()
            # elif token_str == '</story>':
            #     pass
            else:
                print(token_str, end="", flush=True)
            new_context = self.get_context_window(encoding)
            input_tensor = torch.LongTensor([new_context]).to(self.device)

        generated_story = tokenizer.decode(encoding)
        return generated_story

    def get_next_character_probs(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(input_tensor)
            probs = torch.exp(out.squeeze(0)[-1])
            return probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=pathlib.Path, help="Path to the model .pt file")
    parser.add_argument("--tokenizer-directory", type=pathlib.Path, default=pathlib.Path('.'), help="Path to the tokenizer")
    parser.add_argument('--tokenizer-name', type=str, default='bpe-bytelevel', help="Name of the tokenizer")
    parser.add_argument('--context-size', type=int, default=512, help="Size of the context window")
    parser.add_argument('--title', type=str, default=None, help="Title of the story")
    parser.add_argument('--prefix', type=str, default=None, help="Prefix of the story")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = 'cpu'  # if not use_cuda else 'cuda'
    logger.info(f'Using device: {device}')

    logger.info('loading tokenizer...')

    vocab_fname = args.tokenizer_directory / "{}-vocab.json".format(args.tokenizer_name)
    merges_fname = args.tokenizer_directory / "{}-merges.txt".format(args.tokenizer_name)

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

    logger.info('loading model...')
    model = torch.load(args.model_path, weights_only=False).to(device)
    model.eval()

    cls = ModelWrapper(model, tokenizer, args.context_size, device)

    logger.info('ready to generate story...')
    title = args.title
    if title is None:
        title = input('Enter title: ')
    prefix = args.prefix
    if prefix is None:
        prefix = input('Enter prefix: ')
    generated_story = cls.generate_story(title, prefix)

    logger.info('story generated')
    logger.info('saving story...')
    fname = title.replace(' ', '_') + '.txt'
    with open(fname, 'w+') as fp:
        fp.write(generated_story)
    logger.info(f'story saved to {fname}')