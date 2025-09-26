from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time, os, requests

# Setup headless browser
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://in.pinterest.com/search/pins/?q=modern%20TV%20hall")

# Scroll to load a bunch of content
for _ in range(10):
    driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
    time.sleep(2)

# Find all pin links (they start with "/pin/")
pin_links = [
    a.get_attribute("href")
    for a in driver.find_elements(By.TAG_NAME, "a")
    if a.get_attribute("href") and "/pin/" in a.get_attribute("href")
]
pin_links = list(set(pin_links))  # dedupe

print(f"Found {len(pin_links)} unique pins.")

# Prepare folder
os.makedirs("modern_tv_hall_images", exist_ok=True)

# Visit each pin and save the high-res image
for i, link in enumerate(pin_links):
    try:
        driver.get(link)
        time.sleep(2)
        img_url = driver.find_element(By.CSS_SELECTOR, "meta[property='og:image']").get_attribute("content")
        if img_url:
            img_data = requests.get(img_url).content
            with open(f"modern_tv_hall_images/img_{i}.jpg", "wb") as f:
                f.write(img_data)
            print(f"Saved: img_{i}.jpg")
    except Exception as e:
        print(f"Skipping {link}: {e}")

driver.quit()
