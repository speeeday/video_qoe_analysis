# need SSIM, # of changes, rebuffer time, minimize startup delay
from selenium import webdriver
from selenium import common
import sys
import time
import os
from subprocess import call
import numpy as np, re, csv, pickle

from constants import *

# Usage: python youtube_video.py --link [video link]
#
# Make sure to provide the video link within quotes "" via the command
# line because the link often contains shell characters in it


class Youtube_Video_Loader:
	def __init__(self):
		self.t_initialize = time.time()
		self.pull_frequency = .5 # how often to look at stats for nerds box (seconds)
		self.early_stop = 6*60+50 # how long before the end of the video to stop (seconds)

		chrome_options = webdriver.ChromeOptions();
		chrome_options.add_argument("--headless")
		chrome_options.add_argument("--enable-blink-features=HTMLImports")
		#chrome_options.add_extension(CHROME_ADBLOCK_LOCATION) doesn't work in headless chrome
		chrome_options.binary_location = CHROME_BINARY_LOCATION
		chrome_options.add_argument("--window-size=2000,3555") # Needs to be big enough to get all the resolutions
		caps = webdriver.common.desired_capabilities.DesiredCapabilities.CHROME
		caps['goog:loggingPrefs'] = {'performance': 'ALL'}
		self.driver = webdriver.Chrome(chrome_options=chrome_options, desired_capabilities=caps)

		self.video_statistics = {}
		self.logfile_dir = "./logs"
		if not os.path.exists(self.logfile_dir):
			call("mkdir {}".format(self.logfile_dir), shell=True)
		self.log_prefix = "youtube_stats_log-"

	def get_rid_of_ads(self):
		# Check to see if there are ads
		try:
			self.driver.find_element_by_css_selector(".video_ads")
		except:
			# no ads
			return
		done = False
		while not done:
			try:
				self.driver.find_element_by_css_selector(".ytp-ad-skip-button").click()
				#print("Pressed skip")
				# we skipped an ad, wait a sec to see if there are more
				time.sleep(1)
			except:
				# check to see if ad case is still covering
				#print("Display: {}".format(self.driver.find_element_by_css_selector(".video-ads").value_of_css_property("display")))
				if self.driver.find_element_by_css_selector(".video-ads").value_of_css_property("display") != "none":
					# there are more ads, potentially not skippable, just sleep
					time.sleep(1)
				else:
					done = True
					return

	def shutdown(self):
		# write all data to file
		for link in self.video_statistics:
			if self.video_statistics[link]["stats"] == []:
				continue
			video_hash = re.search("youtube\.com\/watch\?v=(.+)",link).group(1)
			fn = os.path.join(self.logfile_dir, self.log_prefix + video_hash)
			with open(fn + "-stats.csv", 'w') as f:
				csvw = csv.DictWriter(f, list(self.video_statistics[link]["stats"][0].keys()))
				csvw.writeheader()
				[csvw.writerow(row) for row in self.video_statistics[link]["stats"]]
			pickle.dump(self.video_statistics[link]["metadata"], open(fn + "-metadata.pkl", 'wb'))

		# kill the browser instance
		self.driver.quit()

	def run(self, link):
		""" Loads a video, pulls statistics about the run-time useful for QoE estimation, saves them to file."""
		self.video_statistics[link] = {"stats":[], "metadata": {}}

		try: # lots of things can go wrong in this loop TODO - report errors 
			self.driver.get(link)
			player = self.driver.find_element_by_css_selector("#player-container-inner")
			# Remove ads
			self.get_rid_of_ads()

			# right click to open option for stats for nerds
			actions = webdriver.common.action_chains.ActionChains(self.driver)
			actions.context_click(player) 
			actions.perform()
			# click on stats for nerds
			self.driver.find_element_by_xpath('/html/body/div[3]/div/div/div[6]/div[2]').click()
			# Find the statistics
			stats_for_nerds_i = None
			for i in range(50):
				try:
					if self.driver.find_element_by_xpath('//*[@id="movie_player"]/div[{}]/div/div[1]/div'.format(i)).text == "Video ID / sCPN":
						stats_for_nerds_i = i
						break
				except common.exceptions.NoSuchElementException:
					continue
			if not stats_for_nerds_i:
				self.driver.save_screenshot("went_wrong.png")
				raise ValueError("Couldn't find stats for nerds box.")

			# get video length
			video_length = self.driver.find_element_by_css_selector('.ytp-time-duration').text.split(":")
			video_length = 60 * int(video_length[0]) + int(video_length[1])
			
			
			# get the resolutions
			self.driver.save_screenshot("on_load.png")
			resolutions = []
			self.driver.find_element_by_css_selector('#movie_player > div.ytp-chrome-bottom > div.ytp-chrome-controls > div.ytp-right-controls > button:nth-child(3)').click()
			self.driver.find_element_by_css_selector('#ytp-id-20 > div > div > div:nth-child(5)').click()
			try:
				i=1
				while True:
					print("Looping through resolutions {}".format(i))
					res_text = self.driver.find_element_by_css_selector("#ytp-id-20 > div > div.ytp-panel-menu > div:nth-child({}) > div > div > span".format(i)).text
					if res_text not in ["Auto", ""]:
						resolutions.append(res_text)
					i+=1
			except:
				pass
			print("Resolutions : {}".format(resolutions))
			self.video_statistics[link]["metadata"]["resolutions"] = resolutions


			tick=time.time()
			self.video_statistics[link]["metadata"]["start_wait"] = tick - self.t_initialize
			print("Clicking Player")
			self.driver.save_screenshot("zeroth.png")
			player.click(); 
			self.driver.save_screenshot("first.png")
			player.click()
			self.driver.save_screenshot("second.png")
			print("Going through stats for nerds")
			# pull data from stats for nerds with whatever frequency you want
			stop = False
			while not stop:
				# get video progress
				# Note - this reading process can take a while, so sleeping is not necessarily advised
				t_calls = time.time()
				viewport_frames = self.driver.find_element_by_xpath('//*[@id="movie_player"]/div[{}]/div/div[2]/span'.format(stats_for_nerds_i)).text
				current_optimal_res = self.driver.find_element_by_xpath('//*[@id="movie_player"]/div[{}]/div/div[3]/span'.format(stats_for_nerds_i)).text
				buffer_health = self.driver.find_element_by_xpath('//*[@id="movie_player"]/div[{}]/div/div[11]/span/span[2]'.format(stats_for_nerds_i)).text
				mystery_text = self.driver.find_element_by_xpath('//*[@id="movie_player"]/div[{}]/div/div[15]/span'.format(stats_for_nerds_i)).text
				mtext_re = re.search("s:(.+) t:(.+) b:(.+)-(.+)", mystery_text)
				state = int(mtext_re.group(1)) # 4 -> paused, 5 -> paused&out of buffer, 8 -> playing, 9 -> rebuffering
				video_progress = float(mtext_re.group(2))

				self.video_statistics[link]["stats"].append({
					"viewport_frames": viewport_frames,
					"current_optimal_res": current_optimal_res,
					"buffer_health": buffer_health,
					"state": state,
					"playback_progress": video_progress,
					"timestamp": time.time(),
				})
				# print("VF: {} \n OR : {} \n BH : {} \n MT : {}".format(
				# 	viewport_frames, current_optimal_res, buffer_health, mystery_text))

				# Check to see if video is almost done
				if time.time() > tick + video_length - self.early_stop:
					stop = True
					player.click() # stop the video
				time.sleep(np.maximum(self.pull_frequency - (time.time() - t_calls),.001))

		except Exception as e:
			self.driver.save_screenshot("went_wrong.png")
			print(sys.exc_info())
		finally:
			self.shutdown()

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--link', action='store')
	args = parser.parse_args()

	yvl = Youtube_Video_Loader()
	yvl.run(args.link)

if __name__ == "__main__":
	main()
