## uses temporal data to try and capture qoe metrics
# Qoe metrics are quality, state, buffer level (high vs low)
import glob, pickle, numpy as np, os, re
from constants import *
from helpers import *
import geoip2.database
import matplotlib.pyplot as plt

import tensorflow as tf

class QOE_Classifier:
	def __init__(self):
		self.features_dir = "./features"
		self.metadata_dir = METADATA_DIR
		self.pcap_dir = "./pcaps"
		self.train_proportion = .8
		self.history_length = 100
		self._types = ["twitch", "netflix", "youtube"]
		#self._types = ["twitch"]
		self.type_to_label_mapping = {_type : i for i,_type in enumerate(self._types)}
		self.sessions_to_clean = {t:[] for t in self._types}
		self.max_dl = float(15e6) # maximum download size -- we divide by this value; could make this depend on the application


		self.X = {"train": {}, "val": {}, "all": []}
		self.Y = {"train": {}, "val": {}, "all": []}
		self.metadata = {"train": [], "val": [], "all": []} # contains info about each example, in case we want to sort on these values

		self.data = {_type: [] for _type in self._types}
		self.visual_quality_labels = {"high": 2,"medium":1,"low":0}
		self.buffer_health_labels = {"high": 2,"medium":1,"low":0}
		
		self.label_types = ["quality", "buffer", "state"]
		self.quality_string_mappings = {
			"twitch": {
				"0x0": self.visual_quality_labels["low"],
				"1600x900": self.visual_quality_labels["high"],
				"1664x936": self.visual_quality_labels["high"],
				"852x480": self.visual_quality_labels["medium"],
				"1280x720": self.visual_quality_labels["high"],
				"1920x1080": self.visual_quality_labels["high"],
				"1704x960": self.visual_quality_labels["high"],
				"640x360": self.visual_quality_labels["low"],
				"256x144": self.visual_quality_labels["low"],
				"284x160": self.visual_quality_labels["low"],
			},
			"netflix": {
				"960x540": self.visual_quality_labels["medium"],
				"1280x720": self.visual_quality_labels["high"],
			},
		}

		self.buffer_intervals = {
			"twitch": {
				"low": [0,2],
				"medium": [2,6],
				"high": [6,100000],
			},
			"netflix": {
				"low": [0,5],
				"medium": [5,15],
				"high": [15,100000]
			},
			"youtube": {
				"low": [0,5],
				"medium": [5,15],
				"high": [15,100000]
			}
		}

		# 0 -> bad, 1 -> good
		self.state_labels = {"good": 1, "bad": 0}
		self.video_states = { # paused and play are good, rebuffering is bad
			"twitch": {
				"playing": self.state_labels["good"], 
				"rebuffer": self.state_labels["bad"], 
				"paused": self.state_labels["good"]
			},
			"netflix": {
				"playing": self.state_labels["good"], 
				"paused": self.state_labels["good"], 
				"waiting for decoder": self.state_labels["bad"]
			},
			# 4-> paused, 8-> playing, 14->loading, 9->rebuffer, 5->(i think) paused, low buffer
			# https://support.google.com/youtube/thread/5642614?hl=en for lengthy discussion
			"youtube": {
				"4": self.state_labels["good"], 
				"8": self.state_labels["good"], 
				"14": self.state_labels["good"], 
				"9": self.state_labels["bad"], 
				"5": self.state_labels["bad"]
			}
		}
		self.all_labels = {"quality": self.visual_quality_labels, "buffer": self.buffer_health_labels, "state": self.state_labels}

	def cleanup(self):
		""" Log files, pcaps, etc.. may be accidentally deleted over time. Get rid of these in examples, since we are missing information."""
		for _type in self.sessions_to_clean:
			fn = os.path.join(self.features_dir, "{}-features.pkl".format(_type))
			these_features = pickle.load(open(fn, 'rb'))
			for _id in self.sessions_to_clean[_type]:
				print("Deleting ID {}, Type: {}".format(_id,_type))
				del these_features[_id]
			pickle.dump(these_features, open(fn,'wb'))

	def get_qoe_label(self, example, i, _type):
		def buffer_health_to_label(buffer_health_seconds):
			for quality in self.buffer_intervals[_type]:
				if buffer_health_seconds >= self.buffer_intervals[_type][quality][0] and\
					buffer_health_seconds < self.buffer_intervals[_type][quality][1]:
					return self.buffer_health_labels[quality]

		# i = 0 -> first report; each additional index corresponds to T_INTERVAL seconds
		t_start = float(example["stats_panel"][0]["timestamp"])
		found=False
		for j, report in enumerate(example["stats_panel"]):
			if i * T_INTERVAL <= (float(report["timestamp"]) - t_start):
				found=True
				break
		if not found:
			return None

		report_of_interest = example["stats_panel"][j]
		state = report_of_interest["state"].strip().lower()
		quality = report_of_interest["current_optimal_res"] # visual quality
		buffer_health = float(report_of_interest["buffer_health"].replace("s",""))

		# Convert the raw buffer health to an integer label
		buffer_health_label = buffer_health_to_label(buffer_health)

		if _type == "twitch":
			quality_label = self.quality_string_mappings[_type][quality]
		elif _type == "netflix":
			quality_label = self.quality_string_mappings[_type][quality]
		elif _type == "youtube":
			# youtube tells you what the maximum quality is, so we treat it differently
			try:
				experienced,optimal = quality.split("/")
			except:
				print(quality);exit(0)
			experienced = [int(el) for el in experienced.split("@")[0].split("x")][1]
			# optimal = [int(el) for el in optimal.split("@")[0].split("x")][1]
			# TODO -- how do I handle cases in which the maximum offered quality is low?
			# this might throw off things and its the exception, so maybe nothing
			# case by case, since quality is this non-linear thing
			# <=360 -> low, 360 - 720 -> medium, >= 720-> high
			if experienced <= 360:
				quality_label = self.visual_quality_labels["low"]
			elif experienced > 360 and experienced < 720:
				quality_label = self.visual_quality_labels["medium"]
			elif experienced >= 720:
				quality_label = self.visual_quality_labels["high"]
			if "c" in state:
				# I don't know what this means, it might be a bug
				# regardless, throw away the data since its only a small fraction of reports
				return None
		else:
			raise ValueError("Type {} not implemented in labeling.".format(_type))

		state_label = self.video_states[_type][state]

		label = [quality_label, buffer_health_label, state_label]
		return label

	def train_and_evaluate(self, label_type):
		cat_y_train,cat_y_val = map(tf.keras.utils.to_categorical,[self.Y["train"], self.Y["val"]])
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Conv2D(
			5,
			(5,5),
			strides=(1,1),
			padding='valid',
			activation=tf.keras.activations.relu,
			use_bias=True,
		))
		model.add(tf.keras.layers.Conv2D(
			3,
			(1,5),
			strides=(1,3),
			padding='valid',
			#activation=tf.keras.activations.relu,
			use_bias=True,
		))
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dropout(rate=.15))
		model.add(tf.keras.layers.Dense(
			3,
			activation=tf.keras.activations.softmax,
			use_bias=True,
		))
		model.compile(optimizer=tf.keras.optimizers.Adam(
			learning_rate=.01
		), loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit(self.X["train"], cat_y_train, batch_size=64,validation_data=(self.X["val"], cat_y_val), epochs=40)
		preds = model.predict(self.X["val"])
		preds = np.argmax(preds, axis=1)
		print(tf.math.confusion_matrix(self.Y["val"], preds, 3))

	def make_train_val(self):
		# Creates self.X (train, val) and self.Y (train, val)
		for _type in self.data:
			for example in self.data[_type]:
				# populate features
				# get the asn list at each point in time
				try:
					raw_byte_stats = pickle.load(open(os.path.join(self.pcap_dir, "{}_processed.pkl".format(example["_id"])),'rb'))
				except FileNotFoundError:
					print("Type: {}, ID: {} not found".format(_type,example["_id"]))
					self.sessions_to_clean[_type].append(example["_id"])
					continue
				byte_stats = example["byte_statistics"]
				size_byte_stats = byte_stats.shape
				for i in range(size_byte_stats[1]):
					# if i % 20 != 0:
					# 	continue
					# get the ips who have been communicated with up until this point (i.e. the IPs that we know about, enforcing causality)
					all_flows = []
					for dst_ip in raw_byte_stats[0]:
						for src_port in raw_byte_stats[0][dst_ip]:
							if sum(raw_byte_stats[0][dst_ip][src_port][0:i] + sum(raw_byte_stats[1][dst_ip][src_port][0:i])) > 0:
								all_flows.append((dst_ip,src_port))
					if all_flows == []: # no data transfer at all yet
						continue
					asns = {}
					with geoip2.database.Reader(os.path.join(self.metadata_dir,"GeoLite2-ASN.mmdb")) as reader:
						for dst_ip,src_port in all_flows:
							try:
								response = reader.asn(dst_ip)
							except geoip2.errors.AddressNotFoundError:
								continue
							asn = response.autonomous_system_organization

							try:
								asns[asn] += 1 # could weight by the amount of traffic
							except KeyError:
								asns[asn] = 1
					n_asns = len(asns)
					asn_dist = get_asn_dist(asns)
					ip_likelihood = get_ip_likelihood(list(set([el[0] for el in all_flows])), _type)
					
					
					temporal_image = np.zeros((size_byte_stats))	
					# only reveal up to the current timestep
					temporal_image[:,-i:,:] = byte_stats[:,0:i,:]
					temporal_image = temporal_image[:,-self.history_length:,:]
					if np.max(np.max(np.max(temporal_image))) == 0: # TODO - this shouldn't occur
						continue
					# normalize between 0 and 1, cheating here; perhaps find a reasonable number to divide by for all files
					# could max out byte counts for unexpectedly high counts
					temporal_image /= self.max_dl

					other_features = [el for el in ip_likelihood]
					[other_features.append(el) for el in asn_dist]
					other_features.append(n_asns)

					# label for this time step
					lab = self.get_qoe_label(example,i, _type)
					if lab is None:
						continue

					self.X["all"].append([temporal_image, other_features])
					self.Y["all"].append(lab)
					self.metadata["all"].append((i, _type)) # timestep, type

		self.X["train"], self.Y["train"], self.X["val"], self.Y["val"] = get_even_train_split(
			self.X["all"], self.Y["all"], self.train_proportion)

	def save_train_val(self):
		# saves the training and validation sets to pkls
		# each label type gets its own train + val set, since each will get its own classifier (at least for now)
		for i,label in enumerate(self.label_types):
			train = {"X": self.X["train"][label], "Y": self.Y["train"][label]}
			val = {"X": self.X["val"][label], "Y": self.Y["val"][label]}
			t_fn, v_fn = os.path.join(self.features_dir, "{}-{}.pkl".format(label,"train")), os.path.join(self.features_dir,"{}-{}.pkl".format(label,"val"))
			pickle.dump(train, open(t_fn,'wb'))
			pickle.dump(val, open(v_fn,'wb'))

	def load_data(self, dtype='raw', label_type=None):
		if dtype == 'raw':
			print("Loading raw data")
			for features_file in glob.glob(os.path.join(self.features_dir, "*-features.pkl")):
				features_type = re.search("{}/(.+)-features.pkl".format(self.features_dir), features_file).group(1)
				if features_type not in self._types:
					continue
				features = pickle.load(open(features_file,'rb'))
				for _id, v in features.items():
					v["_id"] = _id
					self.data[features_type].append(v)
		elif dtype == 'formatted':
			print("Loading pre-formatted data")
			t_fn, v_fn = os.path.join(self.features_dir, "{}-{}.pkl".format(label_type,"train")),\
				os.path.join(self.features_dir,"{}-{}.pkl".format(label_type,"val"))
			t, v = pickle.load(open(t_fn,'rb')), pickle.load(open(v_fn,'rb'))
			self.X["train"], self.Y["train"] = np.array([el[0] for el in t["X"]]), np.array(t["Y"])
			self.X["val"], self.Y["val"] = np.array([el[0] for el in v["X"]]), np.array(v["Y"])
		else:
			raise ValueError("dtype {} not understood".format(dtype))

	def run(self, label_type):
		if not os.path.exists(os.path.join(self.features_dir, "{}-train.pkl".format(label_type))):
			# only do this if we need to
			self.load_data('raw')
			self.make_train_val()
			self.save_train_val()
		self.load_data('formatted', label_type)
		self.train_and_evaluate(label_type)

		self.cleanup()

def main():
	qoec = QOE_Classifier()
	qoec.run('quality')

if __name__ == "__main__":
	main()