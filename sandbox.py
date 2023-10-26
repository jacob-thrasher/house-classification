from crawler import ZillowScraper
import json
from time import sleep

with open('config.json', 'r') as f:
    config = json.load(f)

pfpath = ""
scraper = ZillowScraper(executable_path=config['driver'], profile_path=pfpath)

scraper.search('98367')
sleep(5)

scraper.get_homes()

# link = 'https://photos.zillowstatic.com/fp/2af66996f978e6217d566bd37447ef24-p_e.webp'

