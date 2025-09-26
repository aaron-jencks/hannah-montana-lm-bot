import argparse
import json
import logging
import pathlib
import random
import re
import time
from typing import List, Tuple, Optional, Dict, Any

from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.expected_conditions import presence_of_element_located
from selenium.webdriver.support.wait import WebDriverWait


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PAGE_TIMEOUT = 10


def create_browser() -> webdriver.Chrome:
    options = Options()
    # options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    return driver


def format_url(base_url: str, page: int = 1) -> str:
    return f"{base_url}&p={page}"


def get_page_count(base_url: str) -> int:
    driver = create_browser()
    driver.get(base_url)
    element = WebDriverWait(driver, PAGE_TIMEOUT).until(
        presence_of_element_located(
            (
                By.CSS_SELECTOR,
                "#content_wrapper_inner > center:nth-child(9) > a:nth-child(6)"
            )
        )
    )
    url = element.get_attribute("href")
    driver.close()
    m = re.search(r'&p=(?P<count>\d+)', url)
    if m is None:
        raise Exception(f"Could not parse page count for {url}")
    return int(m.group("count"))


def parse_story_list(base_url: str, page: int) -> List[Tuple[str, str]]:
    driver = create_browser()
    driver.get(format_url(base_url, page))
    WebDriverWait(driver, PAGE_TIMEOUT).until(
        presence_of_element_located(
            (
                By.CSS_SELECTOR,
                "div.z-list"
            )
        )
    )
    elements = driver.find_elements(By.CSS_SELECTOR, "div.z-list")
    stories = []
    for element in elements:
        info = element.find_element(By.CSS_SELECTOR, "a.stitle")
        title = info.text
        story_url = info.get_attribute("href")
        stories.append((title, story_url))
    driver.close()
    return stories


def parse_story(url) -> Optional[str]:
    driver = create_browser()
    driver.get(url)
    try:
        element = WebDriverWait(driver, PAGE_TIMEOUT).until(
            presence_of_element_located(
                (
                    By.CSS_SELECTOR,
                    "div.storytextp"
                )
            )
        )
    except TimeoutException:
        WebDriverWait(driver, PAGE_TIMEOUT).until(
            presence_of_element_located(
                (By.CSS_SELECTOR, "span.gui_warning")
            )
        )
        logger.warning(f"URL {url} returned not found")
        driver.close()
        return None
    story_text = element.text
    driver.close()
    return story_text


def make_corpus_entry(title: str, story: str) -> str:
    flattened_text = story.replace('\n', ' <br> ')
    flattened_title = title.replace('\n', ' <br> ')
    return f'<document> <title> {flattened_title} </title> <story> {flattened_text} </story> </document>'


class Scraper:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.page = 0
        self.page_count: Optional[int] = None

    def restore_from_dict(self, data: Dict[str, Any]):
        self.page = data['page']
        self.page_count = data['page_count']

    def to_dict(self) -> Dict[str, Any]:
        return {
            'page': self.page,
            'page_count': self.page_count,
        }


def save_stories(
        stories: List[Tuple[str, str]], corpus_path: pathlib.Path,
        metadata: Scraper, metadata_path: pathlib.Path
):
    logger.info(f'Saving {len(stories)} stories')

    with open(corpus_path, "a+") as fp:
        lines = [make_corpus_entry(title, text) for title, text in stories]
        fp.write('\n'.join(lines) + '\n')

    with open(corpus_path, 'r') as fp:
        lines = fp.read().splitlines()
        logger.info(f'There are currently {len(lines)} lines in the corpus')

    metadata_dict = metadata.to_dict()
    with open(metadata_path, "w+") as fp:
        json.dump(metadata_dict, fp)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Scrape the fanfiction from the website')
    ap.add_argument('--mature', action='store_true', help='Include mature stories')
    ap.add_argument('--output', type=pathlib.Path, default=pathlib.Path('./corpus.txt'), help='Output file')
    ap.add_argument('--count', type=int, default=-1, help='Number of stories to scrape')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    ap.add_argument('--checkpoint', type=pathlib.Path, default=pathlib.Path('./scraping-checkpoint.json'), help='Checkpoint file for scraping progress')
    args = ap.parse_args()

    random.seed(args.seed)

    base_url = 'https://www.fanfiction.net/tv/Hannah-Montana/?&srt=1&r=103'
    if args.mature:
        base_url = base_url[:-1]

    state = Scraper(base_url)

    from_scratch = not args.checkpoint.exists()
    if not from_scratch:
        with open(args.checkpoint) as fp:
            checkpoint = json.load(fp)
        state.restore_from_dict(checkpoint)
    else:
        pc = get_page_count(base_url)
        state.page_count = pc
        logger.info(f"Found {pc} fanfiction pages")
        time.sleep(3)

    while state.page < state.page_count:
        state.page += 1
        logger.info(f"Parsing page {state.page}")
        stories = parse_story_list(base_url, state.page)
        # random.shuffle(stories)
        logger.info(f"Found {len(stories)} stories")
        time.sleep(3)
        story_texts = []
        for story in stories:
            title, url = story
            text = parse_story(url)
            if text is None:
                logger.warning(f"Could not retrieve story '{title}'")
                continue
            logger.info(f"Parsed story '{title}': {len(text)} characters")
            story_texts.append((title, text))
            if 0 < args.count == len(story_texts):
                break
            time.sleep(3)
        save_stories(story_texts, args.output, state, args.checkpoint)
        if 0 < args.count == len(story_texts):
            break

    logger.info('Scraping complete!')
