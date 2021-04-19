import os
import time
import json
import datetime
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, \
    ElementClickInterceptedException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm

from utils import check_exists_by_class, remove_emojies, save_json_metadata

COOKIES_BT_CSS_SEL = "body > onl-root > div.sidenav-wrapper > div.sidenav-content.takeover-base.onl-allow-takeover-click > div.container > div > onl-cookie > div > div > div > div.cookies__right > a.button.button--large.button--primary.button--expanded.button--noborder"
WAIT_TIMEOUT = 5

def get_page_url(base_url, max_pages=100):
    for i in range(1, max_pages + 1):
        yield f"{base_url}?stran={i}"

def load_all_comments(driver, element):

    el_click_intercept_cnt = 0
    while check_exists_by_class(element, "comments__more", timeout=5):
        try:
            WebDriverWait(driver, WAIT_TIMEOUT).until(EC.presence_of_element_located((By.CLASS_NAME, "comments__more")))
            load_more_bt = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#onl-article-comments > div > div.comments > div.comments__more > button')))
            load_more_bt.click()
            time.sleep(0.5)
            # ActionChains(driver).move_to_element(load_more_bt).click(load_more_bt).perform()
        except (StaleElementReferenceException, TimeoutException, ElementClickInterceptedException) as e:
            if isinstance(e, ElementClickInterceptedException):
                el_click_intercept_cnt += 1

            if el_click_intercept_cnt >= 5:
                return False
            print("Missing element load more. Continuing...", e)
    return True

def get_comment_data(comment_el):
    text = comment_el.find_element_by_class_name("comment__body").text.strip()
    vote_elements = comment_el.find_elements_by_class_name("icon-text--vote")
    votes = [el.find_element_by_class_name("icon-text__value").text for el in vote_elements]
    upvotes = int(votes[0])
    downvotes = int(votes[1])
    user = comment_el.find_element_by_class_name("comment__author").text.strip()
    return {"user": user,
            "content": remove_emojies(text),
            "upvotes": upvotes,
            "downvotes": downvotes}

def parse_comments(comment_elements):
    comments = []
    i = 0
    while i < len(comment_elements):
        if "comment--reply" in comment_elements[i].get_attribute("class"):
            replies = []
            while i < len(comment_elements) and ("comment--reply" in comment_elements[i].get_attribute("class")):
                replies.append(get_comment_data(comment_elements[i]))
                i += 1
            comments[len(comments) - 1]["replies"] += replies
        else:
            comment_data = get_comment_data(comment_elements[i])
            comment_data["replies"] = []
            comments.append(comment_data)
            i += 1

    return comments

def scrape_article_comments(driver, article_url):
    driver.get(article_url)

    # Wait for article body to load, since if it loads later than comments it can interrupt clicks
    WebDriverWait(driver, WAIT_TIMEOUT).until(EC.presence_of_element_located((By.CLASS_NAME, "article__body")))
    WebDriverWait(driver, WAIT_TIMEOUT).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "article__box")))

    # Wait for comments to load and get the element
    try:
        comments_container = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.presence_of_element_located((By.CLASS_NAME, "comments")))
    except TimeoutException:
        print("No comment field found.")
        return None # No comment field was found

    # Click on all the "load more" buttons to load all the comments (if returns false due to scrapping failure, return None
    if not load_all_comments(driver, comments_container):
        return None
    # Get the comments again
    comments_container = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.presence_of_element_located((By.CLASS_NAME, "comments")))

    # Get the comments from the container and remove the element for adding comments
    comment_elements = [comment for comment in comments_container.find_elements_by_class_name("comment") if "comment--add" not in comment.get_attribute("class")]

    if len(comment_elements) == 0:
        return None

    comments = parse_comments(comment_elements)
    # Fetch metadata
    title = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.element_to_be_clickable((By.CLASS_NAME, 'article__title'))).text.strip()
    summary = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.element_to_be_clickable((By.CLASS_NAME, 'article__summary'))).text.strip()
    date_str = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.element_to_be_clickable((By.CLASS_NAME, 'article__info'))).text.split("|")[0].split(",")[1].strip()
    date = datetime.datetime.strptime(date_str, "%d.%m.%Y").isoformat()

    return {
        "url": article_url,
        "title": title,
        "summary": summary,
        "posted_on": date,
        "comments": comments
    }

def get_already_scrapped_urls(dir):
    urls = []
    for filename in os.listdir(dir):
        with open(os.path.join(dir, filename), "r") as f:
            file = json.load(f)
            urls.append(file["url"])

    return set(urls)

if __name__ == '__main__':
    base_url = "https://www.24ur.com/arhiv"
    save_dir = "./data/articles_24ur"

    # Get the webdriver
    driver = webdriver.Chrome("./drivers/chromedriver_90.exe")
    driver.maximize_window()

    # Accept the cookies
    driver.get(base_url)
    cookies_bt = driver.find_element_by_css_selector(COOKIES_BT_CSS_SEL)
    cookies_bt.click()

    # Check that directory for saving is created
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    # Get already scrapped URLs
    already_scrapped_urls = get_already_scrapped_urls(save_dir)
    # Used for sequential naming of articles
    article_count = len(os.listdir(save_dir))

    for page_url in get_page_url(base_url, max_pages=100):
        driver.get(page_url)

        # Get timeline container
        timeline = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.presence_of_element_located((By.CLASS_NAME, "timeline")))
        # Get the article URLs
        article_urls = [item.get_attribute("href") for item in timeline.find_elements_by_class_name("timeline__item")]
        # Remove the already scrapped urls
        article_urls = list(set(article_urls) - already_scrapped_urls)

        # Iterate over the articles on one page and scrape the comments
        for article_url in article_urls:
            try:
                article_data = scrape_article_comments(driver, article_url)
            except:
                print(f"Failed scrapping for: {article_url}")
                continue

            # Check if successfully parsed
            if article_data is None:
                continue

            # Save article
            save_json_metadata(save_dir, str(article_count), article_data)
            print(f"[{article_count}] Parsed article: {article_url}")
            article_count += 1
