# general idea
# we want to classify whether certain packets contain video or not
# 1. load video, save packets
# 2. if video is 
# 2a. private: download HAR file, look at url's requested, look for /range (or similar pattern)
# 2b. public: download HAR file, look at content type
from browsermobproxy import Server
import time, sys
from constants import *
from selenium import webdriver
from subprocess import call

call("killall java", shell=True)
time.sleep(1)
try:
	d = {'port': 8090}
	server = Server(path="/home/ubuntu/browsermob-proxy/bin/browsermob-proxy", 
		options=d)
	server.start()
	time.sleep(1)
	proxy = server.create_proxy()
	time.sleep(1)

	firefox_options = webdriver.firefox.options.Options()
	prof = webdriver.firefox.firefox_profile.FirefoxProfile(FIREFOX_PROFILE_LOCATION)
	prof.set_proxy(proxy.selenium_proxy())
	firefox_options.add_argument("--headless")
	firefox_options.binary_location = FIREFOX_BINARY_LOCATION
	firefox_options.add_argument("--window-size=2000,3555")
	driver = webdriver.Firefox(prof, options=firefox_options)

	proxy.new_har("google")
	url="https://www.youtube.com/watch?v=_rL-lFWlnmw"
	driver.get(url)
	time.sleep(5)
	har = proxy.har

	print(set([req["response"]["content"]["mimeType"] for req in har["log"]["entries"]]))
	
except:
	print(sys.exc_info())
finally:
	server.stop()
	time.sleep(2)
	driver.quit()