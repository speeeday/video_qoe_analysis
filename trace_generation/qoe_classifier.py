## uses temporal data to try and capture qoe metrics
# Qoe metrics are quality, state, buffer level (high vs low)
import glob, pickle, numpy as np, os, re
from constants import *
from helpers import *
import geoip2.database
import matplotlib.pyplot as plt

class QOE_Classifier:
	def __init__(self):
		self.features_dir = "./features"
		self.metadata_dir = METADATA_DIR
		self.pcap_dir = "./pcaps"
		self.train_proportion = .5
		self.history_length = 100
		self._types = ["twitch", "netflix", "youtube"]
		self.type_to_label_mapping = {_type : i for i,_type in enumerate(self._types)}
		self.sessions_to_clean = {t:[] for t in self._types}

		self.data = {_type: [] for _type in self._types}
		self.visual_quality_labels = {"high": 2,"medium":1,"low":0}
		self.buffer_health_labels = {"high": 2,"medium":1,"low":0}
		
		self.quality_string_mappings = {
			"twitch": {
				"0x0": self.visual_quality_labels["low"],
				"1600x900": self.visual_quality_labels["high"],
				"852x480": self.visual_quality_labels["medium"],
				"1280x720": self.visual_quality_labels["high"],
				"1920x1080": self.visual_quality_labels["high"],
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
		self.video_states = { # paused and play are good, rebuffering is bad
			"twitch": {"playing": 1, "rebuffer": 0, "paused": 1},
			"netflix": {"playing": 1, "paused": 1, "waiting for decoder": 0},
			# 4-> paused, 8-> playing, 14->loading, 9->rebuffer, 5->(i think) paused, low buffer
			# https://support.google.com/youtube/thread/5642614?hl=en for lengthy discussion
			"youtube": {"4": 1, "8": 1, "14": 1, "9": 0, "5": 0}
		}

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

	def train_and_evaluate(self):
		pass

	def make_train_val(self):
		# Creates self.X (train, val) and self.Y (train, val)
		self.X = {"train": [], "val": [], "all": []}
		self.Y = {"train": [], "val": [], "all": []}
		self.metadata = {"train": [], "val": [], "all": []} # contains info about each example, in case we want to sort on these values

		for _type in self.data:
			lab = self.type_to_label_mapping[_type]
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
					# if i % 5 != 0:
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
					temporal_image /= np.max(np.max(np.max(temporal_image)))

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

		n_total_examples = len(self.X["all"])
		for _type in self._types:
			inds = [i for i,el in enumerate(self.metadata["all"]) if el[1] == _type]
			labs = [self.Y["all"][i] for i in inds]
			print("Type: {}, {} total examples".format(_type,len(inds)))
			str_labs = ["quality", "buffer", "state"]
			for l in range(3):
				x,c = np.unique([el[l] for el in labs],return_counts=True)
				print("{} - {} with counts {}".format(str_labs[l], x,c))
		from sklearn.model_selection import train_test_split
		split_inds = train_test_split(range(n_total_examples), test_size=1-self.train_proportion)
		train_inds, test_inds = split_inds
		self.X["train"] = [self.X["all"][i] for i in train_inds]
		self.Y["train"] = [self.Y["all"][i] for i in train_inds]
		self.metadata["train"] = [self.metadata["all"][i] for i in train_inds]

		self.X["val"] = [self.X["all"][i] for i in test_inds]
		self.Y["val"] = [self.Y["all"][i] for i in test_inds]
		self.metadata["train"] = [self.metadata["all"][i] for i in test_inds]


	def load_data(self):
		for features_file in glob.glob(os.path.join(self.features_dir, "*-features.pkl")):
			features_type = re.search("{}/(.+)-features.pkl".format(self.features_dir), features_file).group(1)
			if features_type not in self._types:
				continue
			features = pickle.load(open(features_file,'rb'))
			for _id, v in features.items():
				v["_id"] = _id
				self.data[features_type].append(v)

	def run(self):
		self.load_data()

		self.make_train_val()

		self.train_and_evaluate()

		self.cleanup()

def main():
	qoec = QOE_Classifier()
	qoec.run()

if __name__ == "__main__":
	main()