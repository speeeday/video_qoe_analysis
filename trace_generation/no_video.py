from selenium import webdriver
from selenium import common
import sys
import time
import os
import numpy as np, csv

from constants import *

class No_Video_Loader:
	def __init__(self):
		self.t_initialize = time.time()
		self.metadata_dir = "./metadata"

		chrome_options = webdriver.ChromeOptions();
		chrome_options.add_argument("--headless")
		chrome_options.binary_location = CHROME_BINARY_LOCATION
		chrome_options.add_argument("--window-size=2000,3555") # Needs to be big enough to get all the resolutions
		caps = webdriver.common.desired_capabilities.DesiredCapabilities.CHROME
		caps['goog:loggingPrefs'] = {'performance': 'ALL'}
		self.driver = webdriver.Chrome(chrome_options=chrome_options, desired_capabilities=caps)

		self.max_time_execute = MAX_TIME
		self.max_time_sleep = 10

	def done_browsing(self):
		if time.time() - self.t_initialize > self.max_time_execute:
			print("Max time reached - exiting.")
			return True
		return False

	def shutdown(self):
		# kill the browser instance
		self.driver.quit()

	def load_available_webpages(self):
		self.webpages = []
		with open(os.path.join(self.metadata_dir, "top_sites.csv")) as f:
			csvr = csv.DictReader(f)
			for row in csvr:
				if not int(row["has_video"]) == 1:
					self.webpages.append(row["url"])

	def run(self):
		""" Randomly visits webpages that don't have video. Waits for random amounts of time on each page, and then visits another page. 
			Does this until max time is reached."""
		self.load_available_webpages()

		try: # lots of things can go wrong in this loop TODO - report errors 
			done = False
			while not done:
				link = "https://www." + np.random.choice(self.webpages) # randomly choose a link to visit
				print("Visiting link : {}".format(link))
				self.driver.get(link)
				# just do nothing for a little while
				time.sleep(np.random.uniform(low = self.max_time_sleep/2, high = self.max_time_sleep))
				# check to see if done
				done = done or self.done_browsing()

		except Exception as e:
			self.driver.save_screenshot("went_wrong.png")
			print(sys.exc_info())
		finally:
			self.shutdown()

def main():
	nvl = No_Video_Loader()
	nvl.run()

if __name__ == "__main__":
	main()
