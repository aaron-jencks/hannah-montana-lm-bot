import argparse
import random
import re
import time
from typing import List, Tuple
from urllib.robotparser import RobotFileParser

from selenium import webdriver
from selenium.webdriver.common.by import By


def format_url(base_url: str, page: int = 1) -> str:
    return f"{base_url}&p={page}"


def get_page_count(driver: webdriver.Chrome) -> int:
    element = driver.find_element(By.CSS_SELECTOR, "#content_wrapper_inner > center:nth-child(9) > a:nth-child(6)")
    url = element.get_attribute("href")
    m = re.search(r'&p=(?P<count>\d+)', url)
    if m is None:
        raise Exception(f"Could not parse page count for {url}")
    return int(m.group("count"))


def parse_story_list(driver: webdriver.Chrome) -> List[Tuple[str, str]]:
    # div.z-list:nth-child(12)
    t = driver.page_source
    elements = driver.find_elements(By.CSS_SELECTOR, "div.z-list")
    stories = []
    for element in elements:
        info = element.find_element(By.CSS_SELECTOR, "a.stitle")
        title = info.text
        story_url = info.get_attribute("href")
        stories.append((title, story_url))
    return stories


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Scrape the fanfiction from the website')
    ap.add_argument('--mature', action='store_true', help='Include mature stories')
    args = ap.parse_args()

    base_url = 'https://www.fanfiction.net/tv/Hannah-Montana/?&srt=1&r=103'
    if args.mature:
        base_url = base_url[:-1]

    driver = webdriver.Chrome()
    driver.get(base_url)

    pc = get_page_count(driver)
    print(f"Found {pc} fanfiction pages")
    driver.close()
    time.sleep(5)

    order = list(range(1, pc))
    random.shuffle(order)

    for page in order:
        page += 1
        print(f"Parsing page {page}")
        url = format_url(base_url, page)
        driver.get(url)
        stories = parse_story_list(driver)
        print(f"Found {len(stories)} stories")
        print(stories)
        break

    driver.close()