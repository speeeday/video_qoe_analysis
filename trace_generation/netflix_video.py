# # need SSIM, # of changes, rebuffer time, minimize startup delay
# from selenium import webdriver
# import sys
# import time
# import os

# # Usage: python netflix_video.py [video link]
# #
# # Make sure to provide the video link within quotes "" via the command
# # line because the link often contains shell characters in it
# #
# # Make sure you have two files present in the same directory as this file:
# #    credentials/netflix_username.txt - username of a valid netflix subscription
# #    credentials/netflix_password.txt - password for the valid netflix subscription

# chrome_options = webdriver.ChromeOptions();
# # chrome_options.add_argument("--no-gpu")
# # chrome_options.add_argument("--no-sandbox")
# # chrome_options.add_argument("--disable-setuid-sandbox")
# chrome_options.add_argument("--headless")
# chrome_options.binary_location = "/usr/bin/google-chrome"


# caps = webdriver.common.desired_capabilities.DesiredCapabilities.CHROME
# caps['goog:loggingPrefs'] = {'performance': 'ALL'}

# driver = webdriver.Chrome(chrome_options=chrome_options, desired_capabilities=caps)

# driver.get("https://www.netflix.com/login")
# assert "Netflix" in driver.title

# os.system("./preserve_logs.sh")

# my_username = open('credentials/netflix_username.txt').read().strip('\n').split('\n')[0]
# my_password = open('credentials/netflix_password.txt').read().strip('\n').split('\n')[0]

# username = driver.find_element_by_id("id_userLoginId")
# username.clear()
# username.send_keys(my_username)

# password = driver.find_element_by_id("id_password")
# password.clear()
# password.send_keys(my_password)

# #driver.find_element_by_xpath('//button').click()

# driver.find_element_by_class_name("btn-submit").click()

# time.sleep(3)

# driver.find_elements_by_class_name("profile-icon")[1].click()

# #print driver.find_element_by_xpath("//input[@data-uia='login-submit-button']")
# #driver.find_element_by_class_name("btn login-button btn-submit btn-small").click()

# time.sleep(5)


# if len(sys.argv) > 1:
#     link = sys.argv[1]
#     print("[INFO]: Starting Netflix Video at link: {}".format(link))
#     driver.get(link)
    
# DISCONNECTED_MSG = 'Unable to evaluate script: disconnected: not connected to DevTools\n'

# while True:
#     if driver.get_log('driver') != [] and driver.get_log('driver')[-1]['message'] == DISCONNECTED_MSG:
#         print('Browser window closed by user')
#         break
#     time.sleep(1)

from selenium import webdriver
from selenium import common
import sys
import time
import os
from subprocess import call
import numpy as np, re, csv, pickle

from constants import *

# Usage: python netflix_video.py --link [video link]
#
# Make sure to provide the video link within quotes "" via the command
# line because the link often contains shell characters in it


class Netflix_Video_Loader:
	def __init__(self, _id):
		self.t_initialize = time.time()
		self.pull_frequency = .5 # how often to look at stats for nerds box (seconds)
		self.early_stop = 10
		self.max_time = MAX_TIME
		chrome_options = webdriver.ChromeOptions();
		#chrome_options.add_argument("--headless") # the netflix extension doesn't work with headless TODO - submit issue to someone?, get firefox working?
		#chrome_options.add_extension(CHROME_ADBLOCK_LOCATION) doesn't work in headless chrome
		chrome_options.binary_location = CHROME_BINARY_LOCATION
		chrome_options.add_argument("--window-size=2000,3555") # Needs to be big enough to get all the resolutions

		caps = webdriver.common.desired_capabilities.DesiredCapabilities.CHROME
		caps['goog:loggingPrefs'] = {'performance': 'ALL'}
		self.driver = webdriver.Chrome(chrome_options=chrome_options,service_args=["--verbose","--log-path=/home/ubuntu/video_qoe_analysis/trace_generation/chrome_log.log"],desired_capabilities=caps)

		# Load credentials
		self.my_username = open('credentials/netflix_username.txt').read().strip('\n').split('\n')[0]
		self.my_password = open('credentials/netflix_password.txt').read().strip('\n').split('\n')[0]

		self._id = _id
		self.video_statistics = {}
		self.logfile_dir = "./logs"
		self.error_report_dir = ERROR_REPORT_DIR
		if not os.path.exists(self.logfile_dir):
			call("mkdir {}".format(self.logfile_dir), shell=True)
		self.log_prefix = "netflix_stats_log_{}-".format(self._id)

		self.login_url = "https://www.netflix.com/login"

	def save_screenshot(self, img_name):
		self.driver.save_screenshot(os.path.join(self.error_report_dir, "netflix_" + img_name))

	def shutdown(self):
		# write all data to file
		for link in self.video_statistics:
			if self.video_statistics[link]["stats"] == []:
				continue
			video_hash = re.search("netflix\.com\/watch\/(.+)",link).group(1)
			fn = os.path.join(self.logfile_dir, self.log_prefix + video_hash)
			with open(fn + "-stats.csv", 'w') as f:
				csvw = csv.DictWriter(f, list(self.video_statistics[link]["stats"][0].keys()))
				csvw.writeheader()
				[csvw.writerow(row) for row in self.video_statistics[link]["stats"]]
			pickle.dump(self.video_statistics[link]["metadata"], open(fn + "-metadata.pkl", 'wb'))

		# kill the browser instance
		self.driver.quit()

	def login(self):
		time.sleep(5)
		self.driver.get(self.login_url)
		username = self.driver.find_element_by_id("id_userLoginId")
		username.clear()
		username.send_keys(self.my_username)

		password = self.driver.find_element_by_id("id_password")
		password.clear()
		password.send_keys(self.my_password)
		self.driver.find_element_by_class_name("btn-submit").click()
		while True: #wait for the page to load
			try:
				self.driver.find_elements_by_class_name("profile-icon")[NETFLIX_PROFILE_INDEX].click()
				break
			except:
				time.sleep(1)

	def done_watching(self, video_progress):
		# check to see if max time has been hit, or if we are close enough to the video end
		if time.time() - self.t_initialize > self.max_time:
			print("Max time reached - exiting.")
			return True
		elif time.time() > video_progress - self.early_stop:
			print("Close enough to video end - exiting.")
			return True
		return False

	def run(self, link):
		""" Loads a video, pulls statistics about the run-time useful for QoE estimation, saves them to file."""
		self.video_statistics[link] = {"stats":[], "metadata": {}}

		try: # lots of things can go wrong in this loop TODO - report errors 
			self.login()

			self.driver.get(link)
			t_since_last_fetched = time.time()
			while self.driver.current_url != link:
				if time.time() - t_since_last_fetched > 10:
					print("Fetching again")
					print("Note link is: {} desired is {}".format(self.driver.current_url, link))
					self.driver.get(link)
					t_since_last_fetched = time.time()
				time.sleep(1)

			#t_since_last_fetched = time.time()
			while True: # wait for page to load
				try:
					player = self.driver.find_element_by_css_selector(".VideoContainer")
					break
				except:
					# if time.time() - t_since_last_fetched > 10:
					# 	self.driver.save_screenshot("long_wait.png")
					# 	print("Fetching again")
					# 	self.driver.get(link)
					# 	t_since_last_fetched = time.time()
					time.sleep(1)
			# video auto-plays
			tick=time.time()
			self.video_statistics[link]["metadata"]["start_wait"] = tick - self.t_initialize

			# open up the stats panel
			from selenium.webdriver.common.keys import Keys
			actions = webdriver.common.action_chains.ActionChains(self.driver)
			print("Sending keys")
			actions.key_down(Keys.ALT).key_down(Keys.CONTROL).key_down(Keys.SHIFT).send_keys("d").key_up(Keys.ALT).key_up(Keys.SHIFT).key_up(Keys.CONTROL).perform();
			
			print("Going through stats for nerds")
			# pull data from stats for nerds with whatever frequency you want
			stop = False
			video_length = None
			while not stop:
				# get video progress
				# Note - this reading process can take a while, so sleeping is not necessarily advised
				t_calls = time.time()
				info = self.driver.find_element_by_css_selector("div.player-info > textarea").get_attribute("value").split("\n")
				current_optimal_res = re.search("Playing bitrate \(a\/v\): (.+) \/ (.+) \((.+)\)", info[18]).group(3)
				if video_length is None:
					video_length = float(info[9].split(":")[1])
				buffer_health = float(re.search("Buffer size in Seconds \(a\/v\): (.+) \/ (.+)", info[23]).group(1))
				state = info[16].split(":")[1]
				video_progress = float(info[8].split(":")[1])
				self.video_statistics[link]["stats"].append({
					"current_optimal_res": current_optimal_res,
					"buffer_health": buffer_health,
					"state": state,
					"playback_progress": video_progress,
					"timestamp": time.time(),
				})

				# Check to see if video is almost done
				if self.done_watching(tick + video_length):
					stop = True
					# pause
					actions = webdriver.common.action_chains.ActionChains(self.driver)
					actions.send_keys(Keys.SPACE).perform()
				time.sleep(np.maximum(self.pull_frequency - (time.time() - t_calls),.001))
		except Exception as e:
			self.save_screenshot("went_wrong_{}.png".format(self._id))
			print(sys.exc_info())
		finally:
			self.shutdown()

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--link', action='store')
	parser.add_argument('--id', action='store')
	args = parser.parse_args()


	nvl = Netflix_Video_Loader(args.id)
	nvl.run(args.link)

if __name__ == "__main__":
	main()
