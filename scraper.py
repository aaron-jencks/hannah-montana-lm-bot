import argparse
import logging
import pathlib
import random
import re
import time
from typing import List, Tuple

from selenium import webdriver
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


def parse_story(url) -> str:
    driver = create_browser()
    driver.get(url)
    element = WebDriverWait(driver, PAGE_TIMEOUT).until(
        presence_of_element_located(
            (
                By.CSS_SELECTOR,
                "div.storytextp"
            )
        )
    )
    story_text = element.text
    driver.close()
    return story_text


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Scrape the fanfiction from the website')
    ap.add_argument('--mature', action='store_true', help='Include mature stories')
    ap.add_argument('--output', type=pathlib.Path, default=pathlib.Path('./corpus.txt'), help='Output file')
    ap.add_argument('--count', type=int, default=-1, help='Number of stories to scrape')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    args = ap.parse_args()

    random.seed(args.seed)

    base_url = 'https://www.fanfiction.net/tv/Hannah-Montana/?&srt=1&r=103'
    if args.mature:
        base_url = base_url[:-1]

    pc = get_page_count(base_url)
    logger.info(f"Found {pc} fanfiction pages")
    time.sleep(5)

    shuffled_pages = list(range(pc))
    random.shuffle(shuffled_pages)

    story_texts = []
    for page in shuffled_pages:
        page += 1
        logger.info(f"Parsing page {page}")
        stories = parse_story_list(base_url, page)
        random.shuffle(stories)
        logger.info(f"Found {len(stories)} stories")
        for story in stories:
            title, url = story
            logger.info(f"Parsing story '{title}'")
            text = parse_story(url)
            story_texts.append((title, text))
            if 0 < args.count == len(story_texts):
                break
            time.sleep(5)
        if 0 < args.count == len(story_texts):
            break

    with open(args.output, 'w+') as fp:
        lines = []
        for title, text in story_texts:
            flattened_text = text.replace('\n', ' <br> ')
            flattened_title = title.replace('\n', ' <br> ')
            lines.append(f'<document> <title> {flattened_title} </title> <story> {flattened_text} </story> </document>')
        s = '\n'.join(lines)
        fp.write(s)
