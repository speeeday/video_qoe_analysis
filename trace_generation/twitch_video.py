from selenium import webdriver
from selenium import common
import sys
import time, json
import os
from subprocess import call, check_output
import numpy as np, re, csv, pickle

from constants import *

# Usage: python twitch_video.py --link [video link]
#
# Make sure to provide the video link within quotes "" via the command
# line because the link often contains shell characters in it


class Twitch_Video_Loader:
	def __init__(self, _id):
		self.t_initialize = time.time()
		self.pull_frequency = .5 # how often to look at stats for nerds box (seconds)
		self.early_stop = 10 # how long before the end of the video to stop (seconds)

		chrome_options = webdriver.ChromeOptions();
		chrome_options.add_argument("--headless")
		chrome_options.binary_location = CHROME_BINARY_LOCATION
		chrome_options.add_argument("--window-size=2000,3555") # Needs to be big enough to get all the resolutions
		chrome_options.add_argument("--disable-quic")
		caps = webdriver.common.desired_capabilities.DesiredCapabilities.CHROME
		caps['goog:loggingPrefs'] = {'performance': 'ALL'}
		self.driver = webdriver.Chrome(chrome_options=chrome_options, desired_capabilities=caps)

		self.video_statistics = {}
		self.logfile_dir = "./logs"
		self.error_report_dir = ERROR_REPORT_DIR
		self._id = _id
		if not os.path.exists(self.logfile_dir):
			call("mkdir {}".format(self.logfile_dir), shell=True)
		self.log_prefix = "twitch_stats_log_{}-".format(self._id)
		self.max_ad_wait_interval = 60 # (seconds)
		self.max_time = MAX_TIME

	def save_screenshot(self, img_name):
		self.driver.save_screenshot(os.path.join(self.error_report_dir, "twitch_" + img_name))

	def check_for_mature(self):
		"""Some videos are labeled with an 'only for mature audiences' thing that you have to click to proceed. Click it."""
		try:
			self.driver.find_element_by_css_selector("div.content-overlay-gate__allow-pointers.tw-mg-t-3 > button > div > div").click()
			print("Clicked mature button.")
		except:
			print("Not mature stream.")

	def done_watching(self):
		# max time
		if self.max_time is not None:
			if time.time() - self.t_initialize > self.max_time:
				print("Max time reached - exiting.")
				return True

		if np.random.random() < .99:
			# the following checks are expensive, so only perform them occasionally
			return False

		# check if hosting someone else (untested)
		try:
			self.driver.find_element_by_css_selector("div.channel-root__player-container.tw-pd-b-2 > div > div > div > div > div > a > div > figure")
			print("User is hosting someone else - exiting.")
			# is hosting
			return True
		except:
			pass

		# check if just stopped (untested)
		try:
			self.driver.find_element_by_css_selector("div.follow-panel-overlay.tw-absolute.tw-align-items-start.tw-border-radius-small.tw-c-\
				background-overlay.tw-c-text-overlay.tw-elevation-1.tw-flex.tw-flex-column.tw-pd-1.tw-transition.tw-transition\
				--duration-medium.tw-transition--enter-done.tw-transition__slide-over-top.tw-transition__slide-over-top--enter-done")
			print("User is offline - exiting.")
			return True
		except:
			pass
		return False

	def get_rid_of_ads(self):
		# You can't skip ads in twitch, so just wait until they go away
		t_start = time.time()
		refreshed = False
		while True:
			try:
				title_text = self.driver.find_element_by_css_selector("span.tw-c-text-overlay").text
				if "Hosting" in title_text:
					# stream has ended
					print("Note -- this stream is over and the user is hosting someone else.")
					return True
				if "Squad" in title_text:
					print("Note -- this user is squad streaming with others.")
					return True
				time.sleep(1)
				if time.time() - t_start > self.max_ad_wait_interval:
					print("Note -- max ad wait interval hit... refreshing")
					self.save_screenshot("waiting_for_ads.png")
					refreshed = True
					# some non-conforming page probably
					return False
			except:
				# no ads
				print("Couldn't find any ads, returning")
				return True

	def is_buffering(self, link):
		# heuristics to check if the video is buffering
		# one thing is if all the stats don't change over a long enough time period
		n_to_check = 5
		if len(self.video_statistics[link]["stats"]) < n_to_check:
			# too soon to tell
			return False
		most_recent_reports = self.video_statistics[link]["stats"][-n_to_check:]
		# obviously the timestamp will change so don't count that
		grouped_by_key = [set([el[k] for el in most_recent_reports]) for k in most_recent_reports[0] if k != "timestamp"]
		are_singletons = [len(el) == 1 for el in grouped_by_key]
		if sum(are_singletons) == len(are_singletons): # if all of them only had one value
			return True
		return False


	def shutdown(self):
		# write all data to file
		for link in self.video_statistics:
			if self.video_statistics[link]["stats"] == []:
				print("No stats for {}".format(link))
				continue
			video_hash = re.search("twitch\.tv\/(.+)",link).group(1)
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
			link += "/videos" # past broadcasts are listed here
			"""Twitch doesn't neatly expose things like available bitrates, etc..., so we use other tools to get this."""

			# get most recent video (might be live)
			available_formats = check_output("youtube-dl {} --list-formats --playlist-end 1".format(link), shell=True).decode('utf-8')
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

		# link should be to the profile of a live channel
		# i.e. https://twitch.tv/user_name

		self.video_statistics[link] = {"stats":[], "metadata": {}}

		try: # lots of things can go wrong in this loop TODO - report errors 
			self.driver.get(link)
			self.driver.implicitly_wait(5)
			self.check_for_mature()
			# Remove ads
			while not self.get_rid_of_ads():
				self.driver.get(link)
				self.driver.implicitly_wait(5)

			player = self.driver.find_element_by_css_selector(".persistent-player")

			# click settings
			actions = webdriver.common.action_chains.ActionChains(self.driver)
			actions.move_to_element(player)
			actions.perform()
			self.driver.find_element_by_css_selector('.player-controls__right-control-group > div > div button').click()
			# Find the advanced tab
			adv_i = None
			for i in range(2,5):
				try:
					adv_t = self.driver.find_element_by_css_selector("div.settings-menu-button-component.settings-menu-button-component--forced-dark-theme.tw-root--hover.tw-root--theme-dark >\
						div > div > div > div > div > div:nth-child({}) > button > div > div".format(i))
					t = adv_t.text
					if t == "Advanced":
						adv_i = i
						break
				except:
					pass
			if not adv_i:
				raise ValueError("Couldn't find the Advanced button.")
			# Now click the advanced tab
			self.driver.find_element_by_css_selector("div.settings-menu-button-component.settings-menu-button-component--forced-dark-theme.tw-root--hover.tw-root--theme-dark >\
						div > div > div > div > div > div:nth-child({}) > button > div".format(adv_i)).click()
			# click video stats
			vsb_i = None
			for i in range(3,7): # this index might change randomly -- not entirely sure
				try:
					vsb = self.driver.find_element_by_css_selector("div.settings-menu-button-component.settings-menu-button-component--forced-dark-theme.tw-root--hover.tw-root--theme-dark >\
						div > div > div > div > div > div:nth-child({}) > div > div > input".format(i))
					l = vsb.get_attribute("label") 
					if l == "Video Stats":
						vsb_i = i
						break
				except:
					pass
			if not vsb_i:
				raise ValueError("Couldnt find the video stats button.")
			try:
				self.driver.find_element_by_css_selector("div.settings-menu-button-component.settings-menu-button-component--forced-dark-theme.tw-root--hover.tw-root--theme-dark >\
						div > div > div > div > div > div:nth-child({}) > div > div > input".format(vsb_i)).click()
			except:
				# the panel disappeared
				# click settings
				actions = webdriver.common.action_chains.ActionChains(self.driver)
				actions.move_to_element(player)
				actions.perform()
				self.driver.find_element_by_css_selector('.player-controls__right-control-group > div > div button').click()
				# click the advanced tab
				self.driver.find_element_by_css_selector("div.settings-menu-button-component.settings-menu-button-component--forced-dark-theme.tw-root--hover.tw-root--theme-dark >\
					div > div > div > div > div > div:nth-child(2) > button > div").click()
				self.driver.find_element_by_css_selector("div.settings-menu-button-component.settings-menu-button-component--forced-dark-theme.tw-root--hover.tw-root--theme-dark >\
						div > div > div > div > div > div:nth-child({}) > div > div > label".format(vsb_i)).click()
			tick=time.time()
			self.video_statistics[link]["metadata"]["start_wait"] = tick - self.t_initialize
			# pull data from advanced stats with whatever frequency you want
			stop = False
			find_str = "div.tw-root--theme-dark.tw-root--hover > div > div.simplebar-scroll-content > div > div > table > tbody"
			vsl = self.driver.find_elements_by_css_selector(find_str)
			while not stop:
				# get video progress
				# Note - this reading process can take a while, so sleeping is not necessarily advised
				t_calls = time.time()
				video_stats_text = vsl[0].get_attribute("textContent")
				no_fps_re = re.search("Video Resolution(.+)Display Resolution(.+)Skipped Frames(.+)Buffer Size(.+) sec\.Latency To Broadcaster(.+) sec\.Latency Mode(.+)Playback Rate(.+) KbpsBackend Version(.+)Serving ID(.+)Codecs(.+)Play Session ID(.+)", video_stats_text)
				with_fps_re = re.search("Video Resolution(.+)Display Resolution(.+)FPS(.+)Skipped Frames(.+)Buffer Size(.+) sec\.Latency To Broadcaster(.+) sec\.Latency Mode(.+)Playback Rate(.+) KbpsBackend Version(.+)Serving ID(.+)Codecs(.+)Play Session ID(.+)", video_stats_text)
				# right when it loads there are no codecs
				before_loading_re = re.search("Video Resolution(.+)Display Resolution(.+)Skipped Frames(.+)Buffer Size(.+) sec\.Latency To Broadcaster(.+) sec\.Latency Mode(.+)Playback Rate(.+) KbpsBackend VersionServing ID(.+)Play Session ID(.+)", video_stats_text)
				try:
					if with_fps_re: 
						# playing or paused
						current_optimal_res = with_fps_re.group(1)
						buffer_health = float(with_fps_re.group(5))
						if buffer_health == 0:
							state = "paused"
							fps = 0
						else:
							state = "playing"
							fps = int(with_fps_re.group(3))
						latency_mode = with_fps_re.group(7)
						dropped_frames = int(with_fps_re.group(4))
						bitrate = int(with_fps_re.group(8))
					elif no_fps_re:
						# maybe buffering
						# FPS also doesn't show when the screen doesn't change at all
						# This could happen if the video is motionless, which happens for Twitch
						# More than anything (because people might stop moving the mouse)
						current_optimal_res = no_fps_re.group(1)
						buffer_health = float(no_fps_re.group(4))
						if buffer_health < 1:
							state = "rebuffer"
						else:
							state = "playing"
						latency_mode = no_fps_re.group(6)
						dropped_frames = int(no_fps_re.group(3))
						fps = 0
						bitrate = int(no_fps_re.group(7))
					elif before_loading_re:
						# do nothing, just wait for it to load
						print("No stats at all: {}".format(video_stats_text))
						time.sleep(np.maximum(self.pull_frequency - (time.time() - t_calls),.001))
						continue
					else:
						print("Didn't match either.... {} ".format(video_stats_text))
						continue
					if np.random.random() > .9:
						# Print every now and then
						print("Res : {} Buf health: {} state: {}".format(
							current_optimal_res, buffer_health, state))

					self.video_statistics[link]["stats"].append({
						"current_optimal_res": current_optimal_res,
						"buffer_health": buffer_health,
						"fps": fps,
						"state": state,
						"latency_mode": latency_mode,
						"timestamp": time.time(),
						"dropped_frames": dropped_frames,
						"bitrate": bitrate,
					})
				except:
					# if the video rebuffers mid-loop, the regex will fail
					# just ignore this
					print("Hit an error trying to retrieve stats...")
					print(video_stats_text)
					print(sys.exc_info())

				# Check to see if video is almost done
				if self.done_watching():
					stop = True
					from selenium.webdriver.common.keys import Keys
					# pause
					actions = webdriver.common.action_chains.ActionChains(self.driver)
					actions.move_to_element(player).send_keys(Keys.SPACE)
					actions.perform()
				time.sleep(np.maximum(self.pull_frequency - (time.time() - t_calls),.001))

			# Done watching video, get the bitrates
			# self.get_bitrate_data(link) # twitch videos are really big / slow to download
			# I think the bitrate is provided in the advanced stats panel

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


	tvl = Twitch_Video_Loader(args.id)
	tvl.run(args.link)

if __name__ == "__main__":
	main()
