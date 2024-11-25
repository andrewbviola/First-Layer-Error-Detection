from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

'''
Context:
I needed a way to automatically grab a perfect layer image from a website. The website loads the g-code layer viewer
as some sort of dynamic svg. This is why this is a weird elaborate way of downloading the svg as an image, as 
the website does not natively output the g code layer as a picture. Therefore googling and and stackoverflowing was
utilized - Andrew Viola
'''

def currentLayer(name="layer.png",URL="http://enderwire.local:81/#/preview"):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    driver_path = "chromedriver.exe"

    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    driver.get(URL)

    # Waiting for SVG to load: https://stackoverflow.com/questions/76171426/what-is-the-best-way-to-handle-waiting-for-elements-that-are-not-always-present

    WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.TAG_NAME, "svg")))
    time.sleep(15) # giving time to load elements

    # Removing elements: https://stackoverflow.com/questions/22515012/python-selenium-how-can-i-delete-an-element
    driver.execute_script(
        """
        const element1 = document.querySelector("div[data-v-d7de7ace].row");
        if (element1) {
            element1.parentNode.removeChild(element1);
        }
        """)
    driver.execute_script(
        """
        const element2 = document.querySelector("div.preview-options");
        if (element2) {
            element2.parentNode.removeChild(element2);
        }
        """
    )
    driver.execute_script(
        """
        const element3 = document.querySelector("div.preview-name");
        if (element3) {
            element3.parentNode.removeChild(element3);
        }
        """
    )
    svg_element = driver.find_element(By.CSS_SELECTOR, "svg[data-v-a68aa3a1][data-v-6a050874]")
    svg_element.screenshot(f"./Pictures/{name}")

    driver.quit()