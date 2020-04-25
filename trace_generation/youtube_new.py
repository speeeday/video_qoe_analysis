
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
import sys
import time
import os
# Usage: python netflix_video.py [video link]
#
# Make sure to provide the video link within quotes "" via the command
# line because the link often contains shell characters in it
#
# Make sure you have two files present in the same directory as this file:
#    credentials/netflix_username.txt - username of a valid netflix subscription
#    credentials/netflix_password.txt - password for the valid netflix subscription
opts = Options()
opts.log.level = "trace"
#executable_path = './geckodriver'
# create a new firefox session
#driver = webdriver.Firefox(executable_path=executable_path,options = opts)
#driver.implicitly_wait(30)
#driver.maximize_window()

#driver.get("https://www.youtube.com")
#assert "YouTube" in driver.title

#print (driver.current_url)
#sys.exit()

driver = webdriver.Chrome(executable_path = './chromedriver',service_args=["--verbose","--log-path=./chromedriver.log"])
print("test2")
driver.get("https://www.youtube.com/")
print("test3")
assert "YouTube" in driver.title

#print driver.current_url
#sys.exit()

os.system('./preserve_logs.sh')
time.sleep(2)

if len(sys.argv) > 1:
    link = sys.argv[1]
else:
    link = "https://www.youtube.com/watch?v=_qyw6LC5pnE" # place video link here

print "[INFO]: Starting Youtube Video at link: {}".format(link)
driver.get(link)
    
DISCONNECTED_MSG = 'Unable to evaluate script: disconnected: not connected to DevTools\n'

while True:
    driver_log = driver.get_log('driver')
#    print len(driver_log)
    if driver_log != []:
        if driver_log[-1]['message'] == DISCONNECTED_MSG:
            print 'Browser window closed by user'
            break
    time.sleep(1)
