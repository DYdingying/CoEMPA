from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

# Parse Baidu Baike page
def parse_baidu_baike(driver):
    try:
        # Try to get the main content
        # main_content = driver.find_element(By.CLASS_NAME, "mainContent_wtdcH")
        main_content = driver.find_elements(By.CSS_SELECTOR, '[class*="mainContent_"]')[0]
        result = main_content.text.strip()
        if result:
            return result
    except:
        pass

    # If main content does not exist, try to get the first search result
    try:
        # Search result list
        # first_result = driver.find_element(By.CLASS_NAME, "title_OwmZB")
        first_result = driver.find_elements(By.CSS_SELECTOR, '[class*="title_"]')[0]
        first_result_text = first_result.text.strip()
        first_result_link = first_result.get_attribute("href")
        # result = f"First search result: {first_result_text} ({first_result_link})"
        # Navigate to the first search result page
        driver.get(first_result_link)
        # main_content = driver.find_element(By.CLASS_NAME, "mainContent_wtdcH")
        # Try to get the main content again
        main_content = driver.find_elements(By.CSS_SELECTOR, '[class*="mainContent_"]')[0]
        result = main_content.text.strip()
    except:
        # If still fails to get content, return prompt message
        result = "Entry content or search result not found"
    return result

# RAG retrieval function: Baidu Baike augmented generation
def retrieval_augmented_generation(query):
    chrome_driver_path = r"D:\chromedriver-win64\chromedriver.exe"
    # Initialize Chrome browser options
    options = Options()
    options.add_argument("--headless=new") # Headless mode, do not display browser window
    options.add_argument("--window-size=1920,1080")# Set browser window size
    # Set ChromeDriver service
    service = Service(chrome_driver_path)
    # Create browser instance
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Open Baidu Baike homepage
        driver.get("https://baike.baidu.com/")
        time.sleep(2)

        # Enter search term and search
        search_box = driver.find_element(By.CLASS_NAME, "searchInput")
        search_box.send_keys(query)
        # Find and click the search button
        submit_button = driver.find_element(By.CLASS_NAME, "lemmaBtn")
        submit_button.click()
        time.sleep(2)  # Wait for page to load

        # Parse the Baike page
        result = parse_baidu_baike(driver)
        return result

    finally:
        # Close the browser regardless of success or failure
        driver.quit()

if __name__ == "__main__":
    query = input("Please enter the content to search: ")
    search_result = retrieval_augmented_generation(query)
    print(search_result)