import os, numpy as np, csv, glob, re, pickle, bisect

from constants import *
from helpers import *
from video_classifier import Video_Classifier_v2

class ABR_Modeler:
	"""Models each service's ABR algorithm."""
	def __init__(self, abr_last_n_bw=3):
		self.features_dir = "./features"
		self.append_tv = True
		self._types = VIDEO_SERVICES[0:1]
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
		self.abr_model_dir = os.path.join(MODEL_DIR, "abr")
		self.abr_model_name = "abr_forest"

		self.vc = Video_Classifier_v2()
		self.abr_last_n_bw = abr_last_n_bw

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
		min_get_request_size = GET_REQUEST_SIZE
		# units of T_INTERVAL -- proximity of stats report to instance of a get request
		# necessary to only count things near get requests, since that is when the ABR
		# algorithm makes a decision based on the current measured parameters
		get_request_leniency = 1
		bh_leniency = 1
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
							if np.abs(t_future - t_now - bh) > bh_leniency: continue
							if _stats_report["state"] == "rebuffer": continue
							if int(_stats_report['fps']) == 0: continue
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

	def load_data(self, service_type=None, data_type='raw'):
		if data_type == 'raw':
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
		elif data_type == 'formatted':
			if service_type is None:
				raise ValueError("If loading formatted data, you must specify a service type.")
			# load old data and append new data to it
			t_fn, v_fn = os.path.join(self.features_dir, "{}-{}-{}.pkl".format(service_type,"abr_model_data","train")),\
				os.path.join(self.features_dir,"{}-{}-{}.pkl".format(service_type,"abr_model_data","val"))
			if not os.path.exists(t_fn) or not os.path.exists(v_fn): 
				print("Warning -- no data to load for ABR Modeler. You need to make the formatted training and validation sets: {} {}".format(t_fn, v_fn))
				return
			t, v = pickle.load(open(t_fn,'rb')), pickle.load(open(v_fn,'rb'))
			for t_x, t_y, t_m in zip(t['X'], t['Y'], t['metadata']):
				self.abr_data[service_type]['X']['train'].append(t_x)
				self.abr_data[service_type]['Y']['train'].append(t_y)
				self.abr_data[service_type]['metadata']['train'].append(t_m)
			for v_x, v_y, v_m in zip(v['X'], v['Y'], v['metadata']):
				self.abr_data[service_type]['X']['val'].append(v_x)
				self.abr_data[service_type]['Y']['val'].append(v_y)
				self.abr_data[service_type]['metadata']['val'].append(v_m)
		else:
			raise ValueError("Service type {} not understood.".format(service_type))

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

			# Save model for use later
			pickle.dump(clf, open(os.path.join(self.abr_model_dir, self.abr_model_name + "_" + _type + ".pkl"),'wb'))
			self.abr_model[_type] = clf

	def load_abr_model(self, _types = None):
		if _types is None:
			_types = self._types
		for t in _types:
			if self.abr_model[t] is None:
				abr_model_path = os.path.join(self.abr_model_dir, self.abr_model_name)
				try:
					self.abr_model[t] = pickle.load(open(abr_model_path + "_" + t + ".pkl",'rb'))
				except FileNotFoundError:
					print("ABR model for {} doesn't exist, ignoring.".format(t))

	def predict_abr(self, _type, features):
		if self.abr_model[_type] is None:
			self.load_abr_model()
		predicted_labels = self.abr_model[_type].predict(features)
		
		ret_vals = []
		for predicted_label in predicted_labels:
			# return the mean bitrate for videos in this interval
			associated_interval = self.bitrate_class_intervals[_type][predicted_label]
			if associated_interval[1] == np.inf:
				ret_vals.append(associated_interval[0])
			else:
				ret_vals.append(np.mean(self.bitrate_class_intervals[_type][predicted_label]))

		return ret_vals

	def save_train_val(self):
		# saves the training and validation sets to pkls
		if self.append_tv:
			for service_type in self.abr_data:
				self.load_data(service_type, data_type = 'formatted')

		for service_type in self.abr_data:
			train = {'X': self.abr_data[service_type]['X']['train'], 'Y': self.abr_data[service_type]['Y']['train'], 'metadata': self.abr_data[service_type]['metadata']['train']}
			val = {'X': self.abr_data[service_type]['X']['val'], 'Y': self.abr_data[service_type]['Y']['val'], 'metadata': self.abr_data[service_type]['metadata']['val']}
			t_fn, v_fn = os.path.join(self.features_dir, "{}-{}-{}.pkl".format(
				service_type, "abr_model_data", "train")), os.path.join(self.features_dir,"{}-{}-{}.pkl".format(
				service_type,"abr_model_data", "val"))
			pickle.dump(train, open(t_fn,'wb'))
			pickle.dump(val, open(v_fn,'wb'))

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

	def create_abr_model(self, skip_preproc=False):
		if not skip_preproc:
			self.load_data('raw')
			self.form_train_val_abr_model()
			self.save_train_val()
		else:
			for service_type in self._types:
				self.load_data(service_type, data_type='formatted')
		self.train_and_evaluate_abr_model()

if __name__ == "__main__":
	abr_m = ABR_Modeler()
	abr_m.create_abr_model(skip_preproc=True)


