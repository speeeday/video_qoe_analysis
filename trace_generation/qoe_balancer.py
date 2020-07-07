import numpy as np, os, bisect, time, csv, glob, re, pickle
from scipy.optimize import minimize as sci_min
from scipy.optimize import LinearConstraint as lin_constraint
from subprocess import call
from constants import *
from helpers import *
# import tensorflow as tf
from video_classifier import Video_Classifier_v2
import matplotlib.pyplot as plt

def alpha_fairness(utilities, alpha):
	# utilities must be a np array
	epsilon = 1e-3
	if alpha == 1: # using l'hopitals
		return np.sum(np.log(utilities+epsilon))
	elif alpha == np.inf: # max-min fairness
		return np.min(utilities)
	return np.sum(np.power(utilities + epsilon, 1 - alpha) / (1 - alpha))

class QOE_Balance_Modeler:
	"""Performs data analysis-type tasks that build models necessary for proper functioning
	 of the QOE_Balancer."""
	def __init__(self):
		self.features_dir = "./features"
		self._types = ["youtube","twitch", "netflix"]
		self.features_data = {t:[] for t in self._types}
		self.abr_data = {t:
			{"X": {
				"all": [],
				"train": [],
				"val": [],
			},
			"Y": {
				"all": [],
				"train": [],
				"val": []
			},
			"metadata": {
				"all": [],
				"train": [],
				"val": [],
			} 
		} for t in self._types}
		self.abr_model = {t:None for t in self._types}
		self.abr_model_dir = "./models/abr"
		self.abr_model_name = "abr_forest"
		self.vc = Video_Classifier_v2()
		self.abr_last_n_bw = 3

		# max resolution from my experiments was WUXGA, since that's what I set the window
		# resolution to
		# we approximate resolutions for various services with well-known/widely used resolutions
		self.base_resolutions = np.array([
			1920, # WUXGA
			1080, # FHD
			720, # HD
			#480,
			#360,
			240,
			#144,
		])
		# maps resolution specific to service to one of the above resolutions
		self.resolution_to_class = { 
			t:{} for t in self._types
		}

		self.bitrate_class_intervals = {
			"twitch":  {
					0: [0,1000],
					1: [1000,2000],
					2: [2000,3000],
					3: [3000,4000],
					4: [4000,6000],
					5: [6000, np.inf]
			}, 
			"youtube": None,
			"netflix": None,
		}

	def find_bw(self, t, t_s,last_n=1):
		# do binary search through the bandwidth time series to find the 
		# bandwidth corresponding to a time slot
		# since abr algorithms likely have a smoothed estimate of the bandwdith, 
		# optinally include bandwidths from previous timeslots as well	
		arr = sorted(list(self.bandwidth_restrictions.keys()))
		if t < arr[0]:
			# dont have bandwidth measurements for all features
			return None
		bs_p = bisect.bisect(arr,t)
		if arr[bs_p-last_n] < t_s:
			# this bandwdith doesn't apply to this experiment
			return None
		times = arr[bs_p-last_n:bs_p]

		return [self.bandwidth_restrictions[_t] for _t in times]

	def form_train_val_abr_model(self):
		# (buffer, bandwidth) -> resolution request for each service
		# ignore the first n seconds in a video session, since these
		# 1. are noisy 
		# 2. might play by different rules 

		ignore_first_n = 15
		# get request detection
		min_get_request_size = 300
		# units of T_INTERVAL -- proximity of stats report to instance of a get request
		# necessary to only count things near get requests, since that is when the ABR
		# algorithm makes a decision based on the current measured parameters
		get_request_leniency = 1
		for _type in self.features_data:
			for example in self.features_data[_type]:
				t_start = float(example["stats_panel"][0]["timestamp"])
				# Look for GET requests
				byte_statistics = example["byte_statistics"]
				# Limit to only video flows
				all_feats = []
				for flow in byte_statistics[0].keys():
					video_ex = example["video_identification_features"][flow]
					all_feats.append((flow, self.vc.get_features(video_ex)))
				all_feats = [el for el in all_feats if el[1]]
				all_labs = self.vc.predict([el[1] for el in all_feats])
				all_flows = [flow for lab, (flow, feat) in zip(all_labs, all_feats) if lab]
				tmp = [{}, {}]
				for i in range(len(tmp)):
					tmp[i] = {flow: byte_statistics[i][flow] for flow in all_flows}
				byte_statistics = tmp
				t_start_recording_offset = example["start_offset"]
				bin_start = int(np.floor(t_start_recording_offset / T_INTERVAL))

				get_request_times = []
				for flow in byte_statistics[0]:
					for i in range(len(byte_statistics[0][flow])):
						# We don't check for max, since the only thing you'd be uploading on 
						# a video flow are GET requests and ACKs
						# there may be several get requestst in one T_INTERVAL as well, so 
						# maxing this would be tricky
						if byte_statistics[0][flow][i] > min_get_request_size:
							get_request_times.append(i - bin_start)
				get_request_times = np.array(get_request_times)

				for i, stats_report in enumerate(example["stats_panel"]):
					t_now = float(stats_report["timestamp"])
					if t_now - t_start < ignore_first_n:
						continue

					# make sure a get request occurs right around now
					this_bin = int((t_now - t_start) / T_INTERVAL)
					if np.sum(np.abs(this_bin - get_request_times) <= get_request_leniency) == 0:
						continue

					bw = self.find_bw(t_now,t_start,last_n = self.abr_last_n_bw)
					if bw is None:
						# Weren't collecting bandwidth information
						continue

					bh = float(stats_report["buffer_health"])
					# get the stats report bh seconds later (to get resolution)
					report_at_requested_resolution = None
					for _stats_report in example["stats_panel"]:
						t_future = float(_stats_report["timestamp"])
						if t_future - t_now > bh:
							report_at_requested_resolution = _stats_report
							break
					if not report_at_requested_resolution:
						# we ran out of stats reports because data collection ended
						continue

					# Craft features object (bandwidths and buffer health)
					features = bw
					features.append(bh)
					features = np.array(features)
					# really we should be mapping this to bitrate (since thats likely)
					# how this stuff works, but we approximate this with resolution
					try:
						bitrate = float(report_at_requested_resolution["bitrate"])
					except KeyError:
						continue
					bitrate_class = self.bitrate_to_class[_type](bitrate)
					
					resolution_class = self.resolution_to_class[_type][report_at_requested_resolution["current_optimal_res"]]
					
					# label = [resolution_class, bitrate_class]
					label = [bitrate_class]
					md = (i,report_at_requested_resolution["current_optimal_res"])

					self.abr_data[_type]["X"]["all"].append(features)
					self.abr_data[_type]["Y"]["all"].append(label)
					self.abr_data[_type]["metadata"]["all"].append(md)
			if len(self.abr_data[_type]["X"]["all"]) == 0:
				del self.abr_data[_type]
				continue

			# # Form train-test split
			train_proportion=.9
			
			# # For discrete classes
			ret = get_even_train_split(self.abr_data[_type]["X"]["all"], 
				self.abr_data[_type]["Y"]["all"], self.abr_data[_type]["metadata"]["all"],
				train_proportion=train_proportion)
			self.abr_data[_type]["X"]["train"] = ret[0][0]
			self.abr_data[_type]["Y"]["train"] = ret[1][0]
			self.abr_data[_type]["X"]["val"] = ret[2][0]
			self.abr_data[_type]["Y"]["val"] = ret[3][0]
			self.abr_data[_type]["metadata"]["train"] = ret[4][0]
			self.abr_data[_type]["metadata"]["val"] = ret[5][0]

			# # For regression problems
			# n_ex = len(self.abr_data[_type]["X"]["all"])
			# print(n_ex)
			# n_train = int(n_ex * train_proportion)
			# inds = {'train': None, 'val': None}
			# inds['train'] = np.random.choice(range(n_ex), 
			# 	size=n_train, replace=False)
			# inds['val'] = get_difference(range(n_ex), inds['train'])
			# for t in inds:
			# 	self.abr_data[_type]["X"][t] = [self.abr_data[_type]["X"]["all"][i] for
			# 		i in inds[t]]
			# 	self.abr_data[_type]["Y"][t] = [self.abr_data[_type]["Y"]["all"][i] for 
			# 		i in inds[t]]
			# 	self.abr_data[_type]["metadata"][t] = [self.abr_data[_type]["metadata"]["all"][i] for 
			# 		i in inds[t]]

	def load_data_train_abr_model(self):
		# Load bandwidth data
		with open(os.path.join(METADATA_DIR, "all_throughput_limitations.csv")) as f:
			csvr = csv.reader(f)
			self.bandwidth_restrictions = list(csvr)
			# more useful representation
			self.bandwidth_restrictions = {float(t): float(bw) for t,bw in self.bandwidth_restrictions}

		for features_file in glob.glob(os.path.join(self.features_dir, "*-features.pkl")):
			features_type = re.search("{}/(.+)-features.pkl".format(self.features_dir), features_file).group(1)
			if features_type not in self._types:
				continue
			features = pickle.load(open(features_file,'rb'))
			for _id, v in features.items():
				v["_id"] = _id
				self.features_data[features_type].append(v)
		self.make_resolution_to_class_mapping()
		self.make_bitrate_to_class_mapping()

	def make_resolution_to_class_mapping(self):
		for _type in self.features_data:
			resolutions_this_type = set([stats_report["current_optimal_res"]
				for example in self.features_data[_type] for stats_report in 
				example["stats_panel"]])
			for resolution in resolutions_this_type:
				# TODO - fill in
				if _type == "youtube":
					ret = resolution.split('x')
					w = ret[0]
				elif _type == "twitch":
					w,h = resolution.split("x")
				elif _type == "netflix":
					w,h = resolution.split('x')
				closest_match = np.argmin(np.abs(float(w)-self.base_resolutions))
				self.resolution_to_class[_type][resolution] = closest_match

	def make_bitrate_to_class_mapping(self):
		self.bitrate_to_class = {}
		for _type in self._types:
			bitrate_to_class = self.bitrate_class_intervals[_type]
			if bitrate_to_class is None:
				# not yet implemented
				continue
			# checks which interval br falls into
			self.bitrate_to_class[_type] = lambda br : [k for k,v in bitrate_to_class.items() if br >= v[0] and br < v[1]][0]

	def train_and_evaluate_abr_model(self):
		# regression, forest, DL (regression is most interesting)
		from sklearn.ensemble import RandomForestClassifier
		from sklearn.metrics import confusion_matrix, accuracy_score
		for _type in self.abr_data:
			print("Problem type: {}, RF Model".format(_type))
			clf = RandomForestClassifier()
			print("Fitting random forest model...")
			clf.fit(self.abr_data[_type]["X"]["train"], self.abr_data[_type]["Y"]["train"])
			print("Computing metrics...")
			y_pred = clf.predict(self.abr_data[_type]["X"]["val"])
			
			conf_mat = confusion_matrix(self.abr_data[_type]["Y"]["val"], y_pred)
			normalized_cf = conf_mat/np.transpose(np.tile(np.sum(conf_mat,axis=1), (conf_mat.shape[1],1)))
			normalized_acc = 1 -sum(normalized_cf[i,j]/conf_mat.shape[0] for i in range(conf_mat.shape[0]) for j in range(conf_mat.shape[1])if i != j)
			print("Conf Mat: {} \n\n Accuracy: {}".format(normalized_cf, normalized_acc))

			pickle.dump(clf, open(os.path.join(self.abr_model_dir, self.abr_model_name + _type + ".pkl"),'wb'))
			self.abr_model[_type] = clf
		

		# Regression doesn't work
		# from sklearn.linear_model import LinearRegression
		# for _type in self.abr_data:
		# 	# resolution makes more sense for regression
		# 	reg_y_train = [float(el[1].split('x')[0]) for el in self.abr_data[_type]["metadata"]["train"]]
		# 	reg_y_val = [float(el[1].split('x')[0]) for el in self.abr_data[_type]["metadata"]["val"]]
		# 	print("Problem type: {}, Regression Model".format(_type))
		# 	clf = LinearRegression()
		# 	print("Fitting regression model...")
		# 	reg = clf.fit(self.abr_data[_type]["X"]["train"], reg_y_train)
		# 	print("Computing metrics...")
		# 	y_pred = reg.predict(self.abr_data[_type]["X"]["val"])
		# 	print("Score: {}, Coefficients: {}".format(reg.score(
		# 		self.abr_data[_type]["X"]["train"], reg_y_train), reg.coef_))
		# 	print("MSE on val set: {}".format(np.mean(np.square(np.abs(y_pred - reg_y_val)))))

			# x = self.abr_data[_type]["X"]["train"]
			# y = self.abr_data[_type]["Y"]["train"]
			# plt.scatter(x,y)
			# plt.xlabel("Bandwidth")
			# plt.ylabel("Bitrate")
			# plt.savefig("{}-bw-br.pdf".format(_type))

	def load_abr_model(self):
		for t in self._types:
			if self.abr_model[t] is None:
				abr_model_path = os.path.join(self.abr_model_dir, self.abr_model_name)
				try:
					self.abr_model[t] = pickle.load(open(abr_model_path + t + ".pkl",'rb'))
				except FileNotFoundError:
					print("ABR model for {} doesn't exist, ignoring.".format(t))

	def predict_abr(self, _type, features):
		if self.abr_model[_type] is None:
			self.load_abr_model()
		predicted_label = self.abr_model[_type].predict(features)[0]
		
		# return the mean bitrate for videos in this interval
		associated_interval = self.bitrate_class_intervals[_type][predicted_label]
		if associated_interval[1] == np.inf:
			return associated_interval[0]
		else:
			return np.mean(self.bitrate_class_intervals[_type][predicted_label])

	def visualize_abr_model(self):
		# Works, but trees are too big/complicated to get a sense of
		from sklearn.tree import export_graphviz
		for _type in self._types:
			if not self.abr_model[_type]:
				model_fn = os.path.join(self.abr_model_dir, self.abr_model_name + _type + ".pkl")
				if not os.path.exists(model_fn):
					continue
				self.abr_model[_type] = pickle.load(open(model_fn, 'rb'))
			for i,tree_in_forest in enumerate(self.abr_model[_type]):
				export_graphviz(tree_in_forest,
					out_file="tree.dot",
	                feature_names=["Bandwidth", "Buffer Health"],
	                filled=True,
	                rounded=True)
				call("dot -Tpng tree.dot -o tree_{}.png".format(i), shell=True)

	def create_abr_model(self):
		self.load_data_train_abr_model()
		self.form_train_val_abr_model()
		self.train_and_evaluate_abr_model()
		#self.visualize_abr_model()

class QOE_Balancer:
	def __init__(self, alpha=1):
		self.qbm = QOE_Balance_Modeler()
		self.available_bandwidth = 8e3 * 1.0 # could set this dynamically in some way, might be interesting
		self.alpha = alpha # fairness parameter
		self.max_buffers = { # observed empirically, seconds
			"twitch": 30,
			"youtube": 125,
			"netflix": 225,
		}

		self.chunk_sizes = { # observed empircally, seconds
			"twitch": 2,
			"youtube": 6,
			"netflix": 15,
		}

		self.projection_intervals = { # seconds
			"twitch": .5, # livestream, decisions occur on fine time granularities
			"youtube": 3, 
			"netflix": 7,
		}

		self.q_funcs = {
			"twitch": self.q_twitch,
			"youtube": self.q_youtube,
			"netflix": self.q_netflix,
		}

		self.set_random_players()
		

	# Quality functions -- maps bitrate to perceived quality
	def q_twitch(self, br):
		return np.power(br, .6) / np.power(6000,.6) # normalize by max bitrate

	def q_youtube(self, br):
		pass

	def q_netflix(self, br):
		pass

	def set_random_players(self, n_players=5):
		self.players = [{
			"bitrate": 500 + 5500*np.random.random(),
			"buffer_health": 1 + 1*np.random.random(),
			"max_buffer": self.max_buffers["twitch"],	
			"projection_interval": self.projection_intervals["twitch"],
			"q": self.q_funcs["twitch"],
			"chunk_size": self.chunk_sizes["twitch"],
			"service_type": "twitch",	
			"mu": .1,
		} for _ in range(n_players)]

# 		# pull random examples from twitch
# 		self.players = []
# 		n_examples = len(self.qbm.features_data["twitch"])
# 		if n_examples == 0:
# 			self.qbm.load_data_train_abr_model()
# 			n_examples = len(self.qbm.features_data["twitch"])
# 		for i in range(n_players):
# 			r_example = np.random.choice(list(range(n_examples)))
# 			r_example = self.qbm.features_data["twitch"][r_example]
# 			r_stats_rep = np.random.choice(list(range(len(r_example["stats_panel"]))))
# 			r_stats_rep = r_example["stats_panel"][r_stats_rep]
# 			self.players.append({
# 				"bitrate": float(r_stats_rep["bitrate"]),
# 				"buffer_health": float(r_stats_rep["buffer_health"]),
# 				"max_buffer": self.max_buffers["twitch"],	
# 				"projection_interval": self.projection_intervals["twitch"],
# 				"q": self.q_funcs["twitch"],
# 				"chunk_size": self.chunk_sizes["twitch"],
# 				"service_type": "twitch",	
# 				"mu": .1, # Probably don't tune this until I'm actually deploying things
# 			})

		# Re-using calculations
		self.calc_cache = {i:{} for i in range(n_players)}

	def set_alpha(self, alpha):
		self.alpha = alpha

	def get_heuristic_allocation(self):
		# goal is to have no rebuffering, so give bandwidth according to current buffer healthse
		eps = 1e-5
		player_buffer_healths = np.array([self.players[i]["buffer_health"] for i in range(len(self.players))])
		initial_guess = np.reshape((1.0 / (player_buffer_healths+eps)) / np.sum(1.0 / (player_buffer_healths+eps)), (len(self.players),1))
		return initial_guess

	def save_fig(self, fig_name):
		plt.grid(True)
		plt.savefig(os.path.join(FIGURE_DIR, fig_name))
		plt.clf()
		plt.close()

	def solve_for_slices(self, initial_guess=None):
		if initial_guess is None:
			initial_guess = self.get_heuristic_allocation()
		# Define constrinats
		sum_to_one = lin_constraint(np.ones((1,len(self.players))), 1,1)
		greater_than_zero = lin_constraint(np.eye(len(self.players)),0,np.inf)
		constraints = [sum_to_one, greater_than_zero]
		# Call optimizer
		optimal_bandwidth = sci_min(self.get_objective, initial_guess, 
			constraints = constraints,
			options={
				"disp":False,
				"eps": .05,
			}
		).x

		optimal_bandwidth *= self.available_bandwidth

		return optimal_bandwidth

	def compare_uniform_random(self):
		n_players = len(self.players)
		random_dists = np.random.random((100000,n_players))
		random_dists = random_dists / np.sum(random_dists,axis=1)[:,np.newaxis]
		greater_than_uniform = np.sum(random_dists > 1./n_players, axis=1)
		x,cdf_x = get_cdf_xy(greater_than_uniform)
		plt.plot(x,cdf_x)
		plt.xlabel("Number of Players with Greater than Uniform Distribution")
		plt.ylabel("CDF of Trials")
		self.save_fig("compare_uniform_random.pdf")


	def get_objective(self, w, verbose=False):
		"""Objective function for maximization -- fairness for a given bandwidth distribution."""

		projection_window = 8 # units of projection_interval -- how far in the future we are projecting
		projected_utilities = np.zeros(len(self.players))
		if verbose:
			print("Bandwidth Allocation: {}".format(w))
		bw = w * self.available_bandwidth
		for i in range(len(self.players)):
			try:
				# the value of this function depends only on the player
				# so we can save computations if bandwidth allocations
				# don't change between iterations
				projected_utilities[i] = self.calc_cache[i][float(bw[i])]
				continue
			except KeyError:
				pass
			bitrate_init = self.players[i]["bitrate"] # current bitrate, estimated from traffic
			buffer_health_init = self.players[i]["buffer_health"] # starting buffer health, estimated from QoE estimator
			max_buffer = self.players[i]["max_buffer"] # depends on the service
			projection_interval = self.players[i]["projection_interval"] # depends on the service
			q = self.players[i]["q"] # quality function, may depend on service or device type
			bandwidth = bw[i] # bandwidth is constant over the future, set by us
			chunk_size = self.players[i]["chunk_size"] # seconds, typical size of a fetched chunk; depends on the service
			service_type = self.players[i]["service_type"]

			# initialize variables
			buffer_health = buffer_health_init
			net_utility = q(bitrate_init)  # init utility
			last_bitrate = bitrate_init
			chunk_requests = {}
			chunk_buffer = []

			# Populate chunks already in the buffer
			for j in range(int(np.ceil(buffer_health_init / chunk_size))):
				chunk_buffer.append({
						"bitrate": bitrate_init, # assume all chunks in the buffer are at the same bitrate
						"duration": chunk_size, # all chunks are the same length
						"start": j * chunk_size,
						"end": (j+1) * chunk_size
					})

			def find_bitrate_playing_now(ind, cb):
				# get bitrate playing at t = projection_interval * ind
				t = ind * projection_interval
				for chunk in cb:
					if t >= chunk["start"] and t < chunk["end"]:
						return chunk["bitrate"]
				# Empty buffer!
				return 0



			for k in range(1,projection_window):
				# any bitrate requested will be played n_intervals_buffered from now
				n_intervals_buffered = int(np.ceil(buffer_health / projection_interval))
				if buffer_health <= max_buffer - chunk_size and len(chunk_requests) == 0: # check to make sure a request for chunks is likely / possible
					# model requested bitrate at time k accordng to buffer health and bandwidth
					# features is last_n_bandwidths, buffer_health
					# just say last n bandwidths is current bandwidth (can fix this later via memory)
					features = list(np.ones(self.qbm.abr_last_n_bw) * bandwidth)
					features.append(buffer_health)
					bitrate_request = self.qbm.predict_abr(service_type, [features])
					if verbose:
						print("Player {} requesting bitrate {}".format(i,bitrate_request))
					#print("Player {} requesting bitrate: {}".format(i,bitrate_request))
					chunk_requests[k] = {
						"bitrate": bitrate_request,
						"duration": chunk_size,
						"size": bitrate_request * chunk_size, # file size
						"start": k,
						"progress": 0
					}
				# Simulate buffer drain
				buffer_health = np.maximum(0, buffer_health - projection_interval)

				# Get bitrate that's playing now
				#print(chunk_buffer)
				bitrate = find_bitrate_playing_now(k, chunk_buffer)

				# Calculate utility
				utility = q(bitrate) - self.players[i]["mu"] * np.abs(q(bitrate) - q(last_bitrate))
				if verbose:
					print("Player: {}, Utility: {}, Bandwidth: {}, bitrate: {}, buffer_health: {}, t: {}".format(
						i,utility,bandwidth,bitrate,buffer_health,k*projection_interval))
				net_utility += utility
				last_bitrate = bitrate # update bitrate memory

				# Update requests and see if any are done downloading
				# split up bandwidth among requests
				done_requests = []
				for request_id in chunk_requests:
					this_request = chunk_requests[request_id]

					this_request["progress"] += (bandwidth / len(chunk_requests)) * projection_interval / this_request["bitrate"]
					if this_request["progress"] >= chunk_size:
						# done downloading -- add it to the buffer and remove it from the requests
						if len(chunk_buffer) > 0:
							if chunk_buffer[-1]["end"] < k*projection_interval:
								start = k*projection_interval
							else:
								start = chunk_buffer[-1]["end"]
						else: # never any chunks loaded
							start = k*projection_interval
						end = start + this_request["duration"]
						chunk_buffer.append({ # pretend this one starts after the last one ends (neglects out of order arrivals)
							"bitrate": this_request["bitrate"],
							"duration": this_request["duration"],
							"start": start,
							"end": end,
							})
						
						buffer_health += this_request["duration"]
						done_requests.append(request_id) 

				for request_id in done_requests:
					del chunk_requests[request_id]
			projected_utilities[i] = net_utility
			self.calc_cache[i][float(bw[i])] = net_utility
		if verbose:
			print("Projected Utilities: {}".format(projected_utilities))
		fairness = alpha_fairness(projected_utilities, self.alpha)
		if verbose:
			print("Fairness: {}".format(fairness))
		# function minimizes, we want to maximize
		return -1 * fairness


if __name__ == "__main__":
	# qoebm = QOE_Balance_Modeler()
	# qoebm.create_abr_model()
	np.random.seed(31415)
	method_types = ["random", "uniform", "heuristic", "theoretical", "optimized"]
	n_iter = 150
	qb = QOE_Balancer()
	for alpha in [1,2,3,np.inf]:
		qb.set_alpha(alpha)		
		n_players = len(qb.players)
		methods = {
			t: [] for t in method_types
		}
		for i in range(n_iter):
			qb.set_random_players()
			if i % 20 == 0:
				print("{} percent done.".format(i * 100.0 / n_iter))
			# Random guess
			random_dist = np.random.random((n_players,1))
			random_dist /= np.sum(random_dist)
			methods["random"].append(qb.get_objective(random_dist))
			# Uniform allocation
			even_distribution = np.ones((len(qb.players),1)) / len(qb.players)
			methods["uniform"].append(qb.get_objective(even_distribution))
			# Heuristic allocation
			heuristic_dist = qb.get_heuristic_allocation()
			methods["heuristic"].append(qb.get_objective(heuristic_dist))
			# Optimal (impossible) allocation
			insane_dist = 400 * np.ones((len(qb.players), 1))
			methods["theoretical"].append((qb.get_objective(insane_dist)))
			# Solve the problem
			dist = np.array(qb.solve_for_slices())
			dist /= np.sum(dist)
			methods["optimized"].append(qb.get_objective(dist))

		# Absolute performances
		for method in methods:
			x,cdf_x = get_cdf_xy(methods[method])
			plt.plot(x,cdf_x,label=method)
		plt.xlabel("Function Value")
		plt.ylabel("CDF of Iterations")
		plt.legend()
		qb.save_fig("balancer_relative_performance_alpha={}.pdf".format(alpha))

		# Performance comparisons between optimal and each method
		for method in methods:
			if method != "uniform":
				deltas = np.array(methods[method]) - np.array(methods["uniform"])
				print("Method: {} deltas: {}".format(method, deltas))
				x,cdf_x = get_cdf_xy(deltas)
				plt.plot(x,cdf_x,label=method)
				plt.xlim([-5,2])
		plt.xlabel("Relative Performance (Legend Method - Uniform)")
		plt.ylabel("CDF of Iterations")
		plt.legend()
		qb.save_fig("balancer_delta_performance_alpha={}.pdf".format(alpha))
