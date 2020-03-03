# need SSIM, # of changes, rebuffer time, minimize startup delay
from selenium import webdriver
from selenium import common
import sys
import time, json
import os
from subprocess import call, check_output
import numpy as np, re, csv, pickle

from constants import *

# Usage: python youtube_video.py --link [video link]
#
# Make sure to provide the video link within quotes "" via the command
# line because the link often contains shell characters in it


class Youtube_Video_Loader:
	def __init__(self, _id):
		self.t_initialize = time.time()
		self.pull_frequency = .5 # how often to look at stats for nerds box (seconds)
		self.early_stop = 10 # how long before the end of the video to stop (seconds)

		self.max_time = MAX_TIME

		chrome_options = webdriver.ChromeOptions();
		#chrome_options.add_argument("--headless")
		#chrome_options.add_extension(CHROME_ADBLOCK_LOCATION) doesn't work in headless chrome
		chrome_options.binary_location = CHROME_BINARY_LOCATION
		chrome_options.add_argument("--window-size=2000,3555") # Needs to be big enough to get all the resolutions
		caps = webdriver.common.desired_capabilities.DesiredCapabilities.CHROME
		caps['goog:loggingPrefs'] = {'performance': 'ALL'}
		self.driver = webdriver.Chrome(chrome_options=chrome_options, desired_capabilities=caps)

		self.video_statistics = {}
		self._id = _id
		self.logfile_dir = "./logs"
		self.error_report_dir = ERROR_REPORT_DIR
		if not os.path.exists(self.logfile_dir):
			call("mkdir {}".format(self.logfile_dir), shell=True)
		self.log_prefix = "youtube_stats_log_{}-".format(self._id)

	def done_watching(self, video_progress):
		# check to see if max time has been hit, or if we are close enough to the video end
		if time.time() - self.t_initialize > self.max_time:
			print("Max time reached - exiting.")
			return True
		elif time.time() > video_progress - self.early_stop:
			print("Close enough to video end - exiting.")
			return True
		return False

	def save_screenshot(self, img_name):
		self.driver.save_screenshot(os.path.join(self.error_report_dir, "youtube_" + img_name))

	def get_rid_of_ads(self):
		# Check to see if there are ads
		try:
			self.driver.find_element_by_css_selector(".video-ads")
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

	def get_bitrate_data(self, link):
		try:
			"""Youtube doesn't neatly expose things like available bitrates, etc..., so we use other tools to get this."""
			available_formats = check_output("youtube-dl {} --list-formats".format(link), shell=True).decode('utf-8')
			available_formats = available_formats.split("\n")[4:]
			d = []
			resolution_to_format = {}
			for row in available_formats:
				fields = row.split('       ')
				if fields == ['']:
					continue
				code = int(fields[0])
				extension = fields[1].strip()
				if extension != "mp4":
					continue
				resolution = fields[2].split(",")[0].strip()
				resolution = resolution.split(" ")
				try:
					re.search("(.+)x(.+)", resolution[0])
					resolution = resolution[0]
					try:
						resolution_to_format[resolution]
						# prefer webm over mp4
						if extension not in ["mp4", "webm"]:
							raise ValueError("Unprepared to handle extension: {}".format(extension))
						#resolution_to_format[resolution] = ("webm", code)					
						resolution_to_format[resolution] = ("mp4", code)
					except KeyError:
						# this is the only format with this resolution so far
						resolution_to_format[resolution] = (extension, code)
				except:
					# audio
					continue

			bitrates_by_resolution = {}
			for resolution in resolution_to_format:
				fmt,code = resolution_to_format[resolution]
				print("Resolution: {}, Format: {}".format(resolution, fmt))
				if os.path.exists("tmp.{}".format(fmt)):
					call("rm tmp.{}".format(fmt), shell=True)
				# download the video
				call("youtube-dl -o tmp.{} -f {} {}".format(fmt, code, link), shell=True) 
				# get the bitrates for this video
				raw_output = check_output("ffmpeg_bitrate_stats -s video -of json tmp.{}".format(fmt), shell=True)
				bitrate_obj = json.loads(raw_output.decode('utf-8'))
				bitrates_by_resolution[resolution] = bitrate_obj["bitrate_per_chunk"]

				call("rm tmp.{}".format(fmt), shell=True)
			# save this to the links metadata file
			video_hash = re.search("youtube\.com\/watch\?v=(.+)",link).group(1)
			fn = os.path.join(self.logfile_dir, self.log_prefix + video_hash)
			if not os.path.exists(fn + "-metadata.pkl"):
				# just create an empty object
				pickle.dump({}, open(fn + "-metadata.pkl",'wb'))
			this_link_metadata = pickle.load(open(fn + "-metadata.pkl",'rb'))
			this_link_metadata["bitrates_by_resolution"] = bitrates_by_resolution
			pickle.dump(this_link_metadata, open(fn + "-metadata.pkl",'wb'))

		except Exception as e:
			print(sys.exc_info())
		finally:
			self.driver.quit()

	def run(self, link):
		""" Loads a video, pulls statistics about the run-time useful for QoE estimation, saves them to file."""
		self.video_statistics[link] = {"stats":[], "metadata": {}}

		try: # lots of things can go wrong in this loop TODO - report errors 
			self.driver.get(link)
			time.sleep(3) # a common error is that the page takes a little long to load, and it cant find the player 
			max_n_tries,i = 5,0
			while True:
				try:
					player = self.driver.find_element_by_css_selector("#player-container-inner")
					break
				except:
					self.driver.get(link)
					time.sleep(5)
					i += 1
				if i == max_n_tries:
					print("Max number of tries hit trying to get the player-container-inner. Exiting.")
					self.save_screenshot("player_container_unable_{}.png".format(self._id))
					return

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
				self.save_screenshot("no_stats_for_nerds_{}.png".format(self._id))
				raise ValueError("Couldn't find stats for nerds box.")

			# get video length
			while True:
				video_length = self.driver.find_element_by_css_selector('.ytp-time-duration').text.split(":")
				try:
					video_length = 60 * int(video_length[0]) + int(video_length[1])
					break
				except:
					actions = webdriver.common.action_chains.ActionChains(self.driver)
					actions.move_to_element(player)  # bring up the video length box again
					actions.perform()

			tick=time.time()
			self.video_statistics[link]["metadata"]["start_wait"] = tick - self.t_initialize
			print("Starting Player")
			player.click(); 
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
				try:
					state = int(mtext_re.group(1)) # 4 -> paused, 5 -> paused&out of buffer, 8 -> playing, 9 -> rebuffering
				except ValueError:
					state = mtext_re.group(1) # c44->?
				video_progress = float(mtext_re.group(2))

				self.video_statistics[link]["stats"].append({
					"viewport_frames": viewport_frames,
					"current_optimal_res": current_optimal_res,
					"buffer_health": buffer_health,
					"state": state,
					"playback_progress": video_progress,
					"timestamp": time.time(),
				})

				# Check to see if video is almost done
				if self.done_watching(tick + video_length):
					stop = True
					player.click() # stop the video
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
	parser.add_argument('--mode', action='store')
	args = parser.parse_args()

	yvl = Youtube_Video_Loader(args.id)
	if args.mode == "run":
		yvl.run(args.link)
	elif args.mode == "get_bitrate":
		yvl.get_bitrate_data(args.link)
	else:
		raise ValueError("Mode {} not recognized.".format(args.mode))

if __name__ == "__main__":
	main()
