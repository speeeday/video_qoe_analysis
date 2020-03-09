## uses temporal data to try and capture qoe metrics
# Qoe metrics are quality, state, buffer level (high vs low)
import glob, pickle, numpy as np, os, re
from constants import *
from helpers import *
import geoip2.database

class QOE_Classifier:
	def __init__(self):
		self.features_dir = "./features"
		self.metadata_dir = METADATA_DIR
		self.pcap_dir = "./pcaps"
		self.train_proportion = .5
		self._types = ["twitch", "netflix", "youtube"]
		self.type_to_label_mapping = {_type : i for i,_type in enumerate(self._types)}

		self.data = {_type: [] for _type in self._types}

	def get_qoe_label(self, example, _type):
		def quality_to_label(quality_string):
			
		def buffer_health_to_label(buffer_health_seconds):

		def state_to_label(state):
			
		report_of_interest = int(np.ceil((example["start_offset"] - example["stats_panel"][0]["timestamp"]) / T_INTERVAL))
		report_of_interest = example["stats_panel"][report_of_interest]
		if _type == "twitch":
			
		elif _type == "netflix":

		else: # youtube
			quality = 
			"viewport_frames": viewport_frames,
			"current_optimal_res": current_optimal_res,
			"buffer_health": buffer_health,
			"state": state,
			"playback_progress": video_progress,
			"timestamp": time.time(),

	def train_and_evaluate(self):


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
					continue
				byte_stats = example["byte_statistics"]
				size_byte_stats = byte_stats.shape
				for i in range(size_byte_stats[1]):
					if i % 5 != 0:
						continue
					# get the ips who have been communicated with up until this point (i.e. the IPs that we know about, enforcing causality)
					all_ips = [ip for ip in raw_byte_stats[0].keys() if sum(raw_byte_stats[0][ip][0:i] + sum(raw_byte_stats[1][ip][0:i])) > 0]
					if all_ips == []: # no data transfer at all yet
						continue
					asns = {}
					with geoip2.database.Reader(os.path.join(self.metadata_dir,"GeoLite2-ASN.mmdb")) as reader:
						for ip in all_ips:
							try:
								response = reader.asn(ip)
							except geoip2.errors.AddressNotFoundError:
								continue
							asn = response.autonomous_system_organization

							try:
								asns[asn] += 1 # could weight by the amount of traffic
							except KeyError:
								asns[asn] = 1
					n_asns = len(asns)
					asn_dist = get_asn_dist(asns)
					ip_likelihood = get_ip_likelihood(all_ips, _type)
					
					
					temporal_image = np.zeros((size_byte_stats))	
					temporal_image[:,0:i,:] # only reveal up to the current timestep
					temporal_image /= np.max(np.max(np.max(temporal_image))) # normalize between 0 and 1

					other_features = [el for el in ip_likelihood]
					[other_features.append(el) for el in asn_dist]
					other_features.append(n_asns)
					
					self.X["all"].append([temporal_image, other_features])

					# label for this time step
					lab = self.get_qoe_label(example, _type)
					self.Y["all"].append(lab)
					self.metadata["all"].append((i, _type)) # timestep, type

		n_total_examples = len(self.X["all"])
		from sklearn.model_selection import train_test_split
		split_inds = train_test_split(range(n_total_examples), test_size=1-self.train_proportion)
		train_inds, test_inds = split_inds
		self.X["train"] = [self.X["all"][i] for i in train_inds]
		self.Y["train"] = [self.Y["all"][i] for i in train_inds]
		self.X["val"] = [self.X["all"][i] for i in test_inds]
		self.Y["val"] = [self.Y["all"][i] for i in test_inds]

	def load_data(self):
		for features_file in glob.glob(os.path.join(self.features_dir, "*-features.pkl")):
			features_type = re.search("{}/(.+)-features.pkl".format(self.features_dir), features_file).group(1)
			features = pickle.load(open(features_file,'rb'))
			for _id, v in features.items():
				v["_id"] = _id
				self.data[features_type].append(v)

	def run(self):
		self.load_data()

		self.make_train_val()

		self.train_and_evaluate()

def main():
	qoec = QOE_Classifier()
	qoec.run()

if __name__ == "__main__":
	main()