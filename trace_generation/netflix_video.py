
from selenium import webdriver
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


driver = webdriver.Chrome()

driver.get("https://www.netflix.com/login")
assert "Netflix" in driver.title

#print driver.current_url
#sys.exit()

os.system('./preserve_logs.sh')
time.sleep(2)

my_username = open('credentials/netflix_username.txt').read().strip('\n').split('\n')[0]
my_password = open('credentials/netflix_password.txt').read().strip('\n').split('\n')[0]

username = driver.find_element_by_id("id_userLoginId")
username.clear()
username.send_keys(my_username)

password = driver.find_element_by_id("id_password")
password.clear()
password.send_keys(my_password)

#driver.find_element_by_xpath('//button').click()

driver.find_element_by_class_name("btn-submit").click()

time.sleep(3)

driver.find_elements_by_class_name("profile-icon")[1].click()

#print driver.find_element_by_xpath("//input[@data-uia='login-submit-button']")
#driver.find_element_by_class_name("btn login-button btn-submit btn-small").click()

time.sleep(5)


if len(sys.argv) > 1:
    link = sys.argv[1]
    print "[INFO]: Starting Netflix Video at link: {}".format(link)
    driver.get(link)
    
DISCONNECTED_MSG = 'Unable to evaluate script: disconnected: not connected to DevTools\n'

while True:
    if driver.get_log('driver') != [] and driver.get_log('driver')[-1]['message'] == DISCONNECTED_MSG:
        print 'Browser window closed by user'
        break
    time.sleep(1)

