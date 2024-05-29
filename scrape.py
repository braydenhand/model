from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import os
# Replace with the actual path to your ChromeDriver
chrome_driver_path = './chromedriver-win64/chromedriver.exe'
# Get the absolute path to ChromeDriver
chrome_driver_path = os.path.abspath(chrome_driver_path)
# Set up ChromeDriver service
service = Service(chrome_driver_path)

# Create a new instance of the Chrome driver
driver = webdriver.Chrome(service=service)
# Step 2: Fetch the webpage content
url = 'https://www.nba.com/stats/team/1610612738/boxscores?SeasonType=Regular+Season&dir=D&sort=GDATE&Season=2019-20'
driver.get(url)

# Step 3: Wait for the content to load (you may need to adjust the waiting time)
driver.implicitly_wait(10)  # Waits for 10 seconds

# Step 4: Parse the rendered page source with BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html.parser')

# Step 5: Follow the same steps to locate the table
# (Use the same code from the previous example)

# Close the driver
driver.quit()




# Step 3: Find the table element
table = soup.select_one('#__next > div.Layout_base__6IeUC.Layout_justNav__2H4H0 > div.Layout_mainContent__jXliI > main > div.MaxWidthContainer_mwc__ID5AG > section.Block_block__62M07.nba-stats-content-block > div > div.Crom_base__f0niE > div.Crom_container__C45Ti.crom-container > table')

if not table:
    raise ValueError("table not found")
    
# Step 4: Extract table rows and columns
headers = []
rows = []

# Extract headers
for th in table.find_all('th'):
    headers.append(th.text.strip())

# Extract rows
for tr in table.find_all('tr')[1:]:  # Skipping the header row
    cells = tr.find_all('td')
    row = [cell.text.strip() for cell in cells]
    rows.append(row)

# Step 5: Convert to DataFrame
df = pd.DataFrame(rows, columns=headers)
df['W/L'] = df['W/L'].replace({'W': 1, 'L': 0})
# Step 6: Save DataFrame to a machine-readable format (CSV)
df.to_csv('19-20.csv', index=False)

print("Table scraped and saved as 'scraped_table.csv'")
