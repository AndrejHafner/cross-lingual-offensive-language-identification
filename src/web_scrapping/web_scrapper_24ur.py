import time
import json

from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, \
    ElementClickInterceptedException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils import check_exists_by_class

COOKIES_BT_CSS_SEL = "body > onl-root > div.sidenav-wrapper > div.sidenav-content.takeover-base.onl-allow-takeover-click > div.container > div > onl-cookie > div > div > div > div.cookies__right > a.button.button--large.button--primary.button--expanded.button--noborder"
WAIT_TIMEOUT = 5

def get_page_url(base_url, max_pages=100):
    for i in range(1, max_pages + 1):
        yield f"{base_url}?stran={i}"

def load_all_comments(driver, element):
    while check_exists_by_class(element, "comments__more", timeout=5):
        try:
            WebDriverWait(driver, WAIT_TIMEOUT).until(EC.presence_of_element_located((By.CLASS_NAME, "comments__more")))
            load_more_bt = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#onl-article-comments > div > div.comments > div.comments__more > button')))
            load_more_bt.click()
        except (StaleElementReferenceException, TimeoutException, ElementClickInterceptedException) as e:
            # TODO: If elementclickintercepted scroll down a bit and try again?
            print("Missing element load more. Continuing...", e)


def scrape_article_comments(driver, article_url):
    driver.get(article_url)

    # Wait for article body to load, since if it loads later than comments it can interrupt clicks
    WebDriverWait(driver, WAIT_TIMEOUT).until(EC.presence_of_element_located((By.CLASS_NAME, "article__body")))
    WebDriverWait(driver, WAIT_TIMEOUT).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "article__box")))
    # Wait for comments to load and get the element
    comments_container = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.presence_of_element_located((By.CLASS_NAME, "comments")))
    # Click on all the "load more" buttons to load all the comments
    load_all_comments(driver, comments_container)
    # Get the comments again
    comments_container = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.presence_of_element_located((By.CLASS_NAME, "comments")))

    comments = comments_container.find_elements_by_class_name("comment")
    print(article_url, " comments: ", len(comments))
    a = 0

if __name__ == '__main__':
    base_url = "https://www.24ur.com/arhiv"

    # Get the webdriver
    driver = webdriver.Chrome("./drivers/chromedriver_90.exe")

    # Accept the cookies
    driver.get(base_url)
    cookies_bt = driver.find_element_by_css_selector(COOKIES_BT_CSS_SEL)
    cookies_bt.click()

    for page_url in get_page_url(base_url):
        driver.get(page_url)

        # Get timeline container
        timeline = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.presence_of_element_located((By.CLASS_NAME, "timeline")))
        # Get the article URLs
        article_urls = [item.get_attribute("href") for item in timeline.find_elements_by_class_name("timeline__item")]

        # Iterate over the articles on one page and scrape the comments
        for article_url in article_urls:
            scrape_article_comments(driver, article_url)
        a = 0