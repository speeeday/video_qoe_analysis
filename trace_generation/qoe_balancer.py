import numpy as np, os, bisect, time, csv, glob, re, pickle
from scipy.optimize import minimize as sci_min
from scipy.optimize import LinearConstraint as lin_constraint
from subprocess import call
from constants import *
from helpers import *
# import tensorflow as tf
from video_classifier import Video_Classifier_v2
from abr_modeler import ABR_Modeler
import matplotlib.pyplot as plt

def alpha_fairness(utilities, alpha):
	# utilities must be a np array
	epsilon = 1e-3
	if alpha == 1: # using l'hopitals
		return np.sum(np.log(utilities+epsilon))
	elif alpha == np.inf: # max-min fairness
		return np.min(utilities)
	return np.sum(np.power(utilities + epsilon, 1 - alpha) / (1 - alpha))

class QOE_Balancer:
	def __init__(self, alpha=1):
		self.abrm = ABR_Modeler()
		self.available_bandwidth = AVAILABLE_BW * 1.0 # could set this dynamically in some way, might be interesting
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
# 		n_examples = len(self.abrm.features_data["twitch"])
# 		if n_examples == 0:
# 			self.abrm.load_data_train_abr_model()
# 			n_examples = len(self.abrm.features_data["twitch"])
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

	def set_players(self, players):
		self.players = []
		for player_obj in players:
			service = player_obj["service"]
			self.players.append({
					"bitrate": player_obj["bitrate"],
					"buffer_health": player_obj["buffer"],
					"max_buffer": self.max_buffers[service],
					"projection_interval": self.projection_intervals[service],
					"q": self.q_funcs[service],
					"chunk_size": self.chunk_sizes[service],
					'service_type': service,
					"mu": .1,
				})
		# Re-using calculations
		self.calc_cache = {i:{} for i in range(len(players))}


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
					features = list(np.ones(self.abrm.abr_last_n_bw) * bandwidth)
					features.append(buffer_health)
					bitrate_request = self.abrm.predict_abr(service_type, [features])
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
