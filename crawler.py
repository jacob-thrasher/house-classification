from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from time import sleep
import urllib.request
import os

class ZillowScraper:
    def __init__(self, executable_path, profile_path='', headless=False):
        options = webdriver.ChromeOptions()
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument("--profile-directory=Default")

        if profile_path != "":
            options.add_argument("--user-data-dir="+profile_path)

        if headless:
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
        
        #Instantiate driver object
        self.chromedriver = webdriver.Chrome(executable_path=executable_path, options=options)
        self.chromedriver.set_window_size(1920, 1080)
        self.chromedriver.get("https://www.zillow.com")

    def bypass_captcha(self):
        button = self.chromedriver.find_element(By.ID, 'px-captcha')
        action = ActionChains(self.chromedriver)
        click = ActionChains(self.chromedriver)
        print('down')
        action.click_and_hold(button)
        action.perform()
        sleep(7)
        action.release(button)
        action.perform()
        print('up')
        sleep(0.2)
        action.release(button)

    def search(self, location: str):
        '''
        navigate to desired location (ie Seattle, Chicago, Harpers Ferry, etc)

        Args:
            location (str): desired city
        '''

        location = location.replace(' ', '-') + '_rb'
        search = f'https://www.zillow.com/homes/{location}'
        self.chromedriver.get(search)

    def get_homes(self):
        houses_header = self.chromedriver.find_element(By.XPATH, '//*[@id="grid-search-results"]/ul')
        house_links = houses_header.find_elements(By.XPATH, '*')
        
        house = house_links[0]
        house.click()
        images = house.find_elements(By.CSS_SELECTOR, 'img')
        image_links = [x.get_attribute('src') for x in images]
        self.download_images(links=image_links, dst=None, prefix='test')

    def download_images(self, links=None, dst=None, prefix=None, filetype='png'):
        '''
        Download list of images to dst

        args:
            links (list) - List of links to images to download
            dst (str): Location to save images
            prefix (str): Filename prefix
            filetype (str): File extension
        '''

        if dst is None:
            dst = os.getcwd()

        count = 1
        for link in links:
            print(link)
            filename = prefix + '_' + str(count) + '.' + filetype
            path = os.path.join(dst, filename)
            urllib.request.urlretrieve(link, path)
            count += 1
        
        