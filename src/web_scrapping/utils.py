import emoji
import json
import os

from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def check_exists_by_xpath(element, xpath):
    try:
        element.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True

def check_exists_by_class(driver, css_class, timeout=1):
    try:
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CLASS_NAME, css_class)))
    except TimeoutException:
        return False
    return True

def save_json_metadata(dir, filename, obj):
    with open(f"{os.path.join(dir, filename)}.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)