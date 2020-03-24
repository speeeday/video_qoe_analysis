# for each of youtube, twitch, netflix, find pages to visit and run data collection programs by watching them
# also run collection sessions for no_video


from selenium import webdriver
from selenium import common
import sys
import time, json
import os
from subprocess import call, check_output
import numpy as np, re, csv, pickle

from constants import *

class Data_Gatherer:
	def __init__(self):
		self.n_to_collect = 10 # number of videos or sessions to visit
		# Load credentials
		self.netflix_login_url = "https://www.netflix.com/login"
		self.netflix_username = open('credentials/netflix_username.txt').read().strip('\n').split('\n')[0]
		self.netflix_password = open('credentials/netflix_password.txt').read().strip('\n').split('\n')[0]

	def startup(self):
		chrome_options = webdriver.ChromeOptions();
		chrome_options.add_argument("--headless")
		chrome_options.binary_location = CHROME_BINARY_LOCATION
		chrome_options.add_argument("--window-size=2000,3555") # Needs to be big enough to get all the resolutions
		caps = webdriver.common.desired_capabilities.DesiredCapabilities.CHROME
		caps['goog:loggingPrefs'] = {'performance': 'ALL'}
		self.driver = webdriver.Chrome(chrome_options=chrome_options, desired_capabilities=caps)

	def shutdown(self):
		# kill the browser instance
		self.driver.quit()

	def call_data_gather(self, _type, link=None):
		if link:
			cmd = "./run_data_collect.sh {} {}".format(_type,link)
		else:
			cmd ="./run_data_collect.sh {}".format(_type) 

		call(cmd,shell=True)

		call("python data_aggregator.py --mode run --type {}".format(_type), shell=True) # to prevent space problems

	def netflix_login(self):
		self.driver.get(self.netflix_login_url)
		username = self.driver.find_element_by_id("id_userLoginId")
		username.clear()
		username.send_keys(self.netflix_username)

		password = self.driver.find_element_by_id("id_password")
		password.clear()
		password.send_keys(self.netflix_password)
		self.driver.find_element_by_class_name("btn-submit").click()
		while True: #wait for the page to load
			try:
				self.driver.find_elements_by_class_name("profile-icon")[NETFLIX_PROFILE_INDEX].click()
				break
			except:
				self.driver.save_screenshot("went_wrong.png")
				time.sleep(1)

	def run(self):
		max_n_iters = 20 # this scrolling method will only reveal up to a certain number of videos (but still ~100)

		# Note -- we kill the selenium driver each time, since the subprocess spawns a different selenium driver


		# print("-----Starting Netflix data gather.------")
		# try: # lots of things can go wrong in this loop TODO - report errors 
		# 	self.startup()
		# 	# Netflix
		# 	# get a list of the n most popular netflix videos 
		# 	self.netflix_login()
		# 	self.netflix_video_ids = []
			
		# 	# we are on the splash page, go through all the videos on the page and get the video IDs
		# 	i = 0
			
		# 	while len(self.netflix_video_ids) < self.n_to_collect and i < max_n_iters:
		# 		all_video_boxes = self.driver.find_elements_by_class_name('slider-item')
		# 		print("Found {} video boxes.".format(len(all_video_boxes)))
		# 		for video_box in all_video_boxes:
		# 			try:
		# 				ptrack_content = video_box.find_element_by_css_selector('.ptrack-content').get_attribute("data-ui-tracking-context")
		# 			except:
		# 				continue
		# 			video_id_re = re.search("\%22video_id\%22\:(.+)\,\%22image_key",ptrack_content)
		# 			video_id = video_id_re.group(1)
		# 			self.netflix_video_ids.append(video_id)
		# 		self.netflix_video_ids = list(set(self.netflix_video_ids)) # remove duplicates
		# 		print("Up to {} netflix video ids.".format(len(self.netflix_video_ids)))
		# 		# scroll the page down to reveal more boxes
		# 		self.driver.execute_script("window.scrollBy(0,700)")
		# 		i += 1
		# except Exception as e:
		# 	print(sys.exc_info())
		# finally:
		# 	self.shutdown()
		# self.netflix_video_ids = [el for el in self.netflix_video_ids if el]
		# # Now gather stats about each of these
		# if len(self.netflix_video_ids) > self.n_to_collect:
		# 	self.netflix_video_ids = np.random.choice(self.netflix_video_ids, self.n_to_collect, replace=False)
		# for video_id in self.netflix_video_ids:
		# 	print("Watching netflix video with id : {}".format(video_id))
		#	self.call_data_gather("netflix","https://www.netflix.com/watch/{}".format(video_id))

		print("-----Starting Youtube data gather.------")
		try:
			self.startup()
			
			# Youtube
			# get a list of youtube videos on the splash page
			self.driver.get("https://www.youtube.com")
			self.youtube_video_links = []

			# we are on the splash page, go through all the videos on the page and get the video IDs
			i = 0
			while len(self.youtube_video_links) < self.n_to_collect and i < max_n_iters:
				all_video_boxes = self.driver.find_elements_by_css_selector('a#thumbnail')
				for video_box in all_video_boxes:
					self.youtube_video_links.append(video_box.get_attribute("href"))
				self.youtube_video_links = list(set(self.youtube_video_links))
				# scroll the page down to reveal more boxes
				self.driver.execute_script("window.scrollBy(0,700)")
				i += 1
		except Exception as e:
			print(sys.exc_info())
		finally:
			self.shutdown()
		
		# Now gather stats about each of these
		self.youtube_video_links = [el for el in self.youtube_video_links if el]
		if len(self.youtube_video_links) > self.n_to_collect:
			self.youtube_video_links = np.random.choice(self.youtube_video_links, self.n_to_collect, replace=False)
		for video_link in self.youtube_video_links:
			print("Conducting data gather for video : {}".format(video_link))
			self.call_data_gather("youtube", video_link)

		print("-----Starting Twitch data gather.------")
		try:
			self.startup()
			# Twitch
			# get a list of popular, live channels
			self.driver.get("https://www.twitch.tv/directory/all")
			self.twitch_profiles = []

			# we are on the splash page, go through all the channels on the page and get the profile names
			i = 0
			while len(self.twitch_profiles) < self.n_to_collect and i < max_n_iters:
				all_video_boxes = self.driver.find_elements_by_css_selector('div.tw-flex-grow-1.tw-flex-shrink-1.tw-full-width.tw-item-order-2.tw-media-card-meta__text-container\
				 > div.tw-media-card-meta__links > div:nth-child(1) > p > a')
				for video_box in all_video_boxes:
					if "directory" not in video_box.get_attribute('href'): # this is a link to a general game directory
						# TODO - check to make sure the channel name is unicode (not foreign)
						self.twitch_profiles.append(video_box.text)
				self.twitch_profiles = list(set(self.twitch_profiles))
				# scroll the page down to reveal more boxes
				self.driver.execute_script("window.scrollBy(0,700)")
				i += 1
		except Exception as e:
			print(sys.exc_info())
		finally:
			self.shutdown()

		## now gather stats about each of these
		print(self.twitch_profiles)
		self.twitch_profiles = [el for el in self.twitch_profiles if el]
		if len(self.twitch_profiles) > self.n_to_collect:
			self.twitch_profiles = np.random.choice(self.twitch_profiles, self.n_to_collect, replace=False)
		for twitch_profile in self.twitch_profiles:
			print("Watching stream of profile : {}".format(twitch_profile))
			self.call_data_gather("twitch", "https://www.twitch.tv/{}".format(twitch_profile))

		print("-----Starting no video data gather.------")
		# No video
		for i in range(self.n_to_collect * 3): # get 3x as many of these sessions, so there is no data imbalance
			self.call_data_gather("no_video")

def main():
	import argparse
	parser = argparse.ArgumentParser()

	dg = Data_Gatherer()
	dg.run()

if __name__ == "__main__":
	main()
