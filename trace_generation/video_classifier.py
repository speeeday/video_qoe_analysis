## uses high level features and a random forest classifier to distinguish video from non-video flows
import glob, pickle, numpy as np, os, re, math
import matplotlib.pyplot as plt
from constants import *
from helpers import *

class Video_Classifier_v2:
	# Classifies individual flows as containing video or not
	# subsequently using an asn lookup on the IP could differentiate between video services trivially
	def __init__(self):
		self.features_dir = "./features"
		self.fig_dir = "./figures"
		self.figure_prefix = ""
		self.model_dir = "./models/video_classifier"
		self.model_name = "video_class_v2.pkl"
		self.model = None
		self.metadata_dir = METADATA_DIR
		self.train_proportion = .8
		self._types = ["twitch", "netflix", "youtube"]
		#self._types = ["netflix"]
		self.type_to_label_mapping = {_type : i for i,_type in enumerate(self._types)}
		self.max_tpt = float(15e6) # maximum download size -- we divide by this value; could make this depend on the application

		self.X = {"train": {}, "val": {}, "all": []}
		self.Y = {"train": {}, "val": {}, "all": []}
		self.metadata = {"train": [], "val": [], "all": []} # contains info about each example, in case we want to sort on these values

		self.label_types = ["video_class"]
		self.data = {_type: [] for _type in self._types}
		self.all_labels = {"video_class": [0,1]}

		# Feature processing
		self.no_feature = -1
		self.tpt_log_base = 4
		self.bytes_log_base = 4
		
	def train_and_evaluate(self):
		from sklearn.ensemble import RandomForestClassifier
		from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
		for p_type in self.X['train']:
			print("Problem type: {}".format(p_type))
			clf = RandomForestClassifier()
			print("Fitting random forest model...")
			clf.fit(self.X["train"][p_type], self.Y["train"][p_type])
			print("Computing metrics...")
			y_pred = clf.predict(self.X["val"][p_type])
			
			
			conf_mat = confusion_matrix(self.Y["val"][p_type], y_pred)
			accuracy = accuracy_score(self.Y["val"][p_type], y_pred)
			prec = precision_score(self.Y["val"][p_type], y_pred)
			rec = recall_score(self.Y["val"][p_type], y_pred)
			f1 = f1_score(self.Y["val"][p_type], y_pred)
			print("Conf Mat: {} \n\n Accuracy: {}".format(conf_mat, accuracy))
			print("Precision: {}, Recall: {}, F1: {}".format(prec, rec, f1))

			# visualize predictions
			self.visualize_data(self.X['val'][p_type], self.Y['val'][p_type], y_pred)

			print("Saving model for later.")
			pickle.dump(clf, open(os.path.join(self.model_dir, self.model_name),'wb'))

	def load_data(self, label_type=None):
		print("Loading raw data")
		for features_file in glob.glob(os.path.join(self.features_dir, "*-features.pkl")):
			features_type = re.search("{}/(.+)-features.pkl".format(self.features_dir), features_file).group(1)
			if features_type not in self._types:
				continue
			features = pickle.load(open(features_file,'rb'))
			for _id, v in features.items():
				v["_id"] = _id
				self.data[features_type].append(v)

	def visualize_data(self, x, y, y_pred):
		# visualize the confusion matrix
		cs = {'TN': 'r', 'TP': 'b', 'FP': 'y', 'FN': 'k'}
		features_of_interest = [0,6]
		axis_labels = ["Bytes", "TLS"]
		arrs = [[_x[f] for _x in x] for f in features_of_interest]
		colors = []
		for _y, _y_pred in zip(y,y_pred):
			if _y == _y_pred and _y:
				colors.append(cs['TP'])
			elif _y == _y_pred and not _y:
				colors.append(cs['TN'])
			elif _y != _y_pred and _y:
				colors.append(cs['FN'])
			else:
				colors.append(cs['FP'])
		arrs = [np.array(_x,dtype=np.float64) + .05*np.random.randn(len(_x)) for _x in arrs]
		plt.scatter(arrs[0],arrs[1], c=colors)
		plt.xlabel("Bytes")
		plt.ylabel("TLS")
		self.save_fig("feature_viz.pdf")

	def get_bytes_feature(self, n_bytes):
		mtu = 1500 # bytes
		return int(math.log(n_bytes / mtu, self.bytes_log_base))

	def get_tpt_feature(self, tpt):
		conversion = 1e3 # bytes / sec
		return int(math.log(tpt / conversion, self.tpt_log_base))

	def get_features(self, flow_obj):
		# For all byte-type features, take log and integerize
		# Total Bytes
		if flow_obj["total_bytes"] == 0:
			return None
		total_bytes = self.get_bytes_feature(flow_obj["total_bytes"])
		# Up
		if flow_obj["total_bytes_up"] > 0:
			tb_up = self.get_bytes_feature(flow_obj["total_bytes_up"])
		else:
			tb_up = self.no_feature
		# Down
		if flow_obj["total_bytes_down"] > 0:
			tb_down = self.get_bytes_feature(flow_obj["total_bytes_down"])
		else:
			tb_down = self.no_feature
		# Take log2 of the throughput variables, since finer granularity provides more information
		try:
			mean_tpt = self.get_tpt_feature(flow_obj["mean_throughput"])
		except KeyError:
			mean_tpt = self.no_feature
		try:
			max_tpt = self.get_tpt_feature(flow_obj["max_throughput"])
		except KeyError:
			max_tpt = self.no_feature
		try:
			if flow_obj["std_throughput"] > 0:
				std_tpt = self.get_tpt_feature(flow_obj["std_throughput"])
			else:
				std_tpt = self.no_feature
		except KeyError:
			std_tpt = self.no_feature


		# TLS hostname
		try:
			if "googlevideo.com" in flow_obj["tls_server_hostname"]:
				tls = 1
			elif "nflxvideo.net" in flow_obj["tls_server_hostname"]:
				tls = 1
			elif "ttvnw.net" in flow_obj["tls_server_hostname"]:
				tls = 1
			else:
				tls = 0
		except KeyError:
			tls = 0

		# FFT harms performance
		# Would probably want something more like distance between peaks or something
		# n_fft_to_keep = 2
		# try:
		# 	fft_peaks = this_ex["peak_fft_i"]
		# 	# FFT should be symmetric, so half the information doesn't matter
		# 	fft_inds = np.abs(np.array(this_ex["peak_fft_i"]) - 32)
		# 	# Keep the top 3
		# 	fft_feats = list(fft_inds[-n_fft_to_keep:])
		# except KeyError:
		# 	fft_feats = [-1] * n_fft_to_keep
		# features = features + fft_feats

		features = [total_bytes, tb_up, tb_down, mean_tpt, max_tpt, std_tpt, tls]

		return features

	def make_train_val(self):
		# Creates self.X (train, val) and self.Y (train, val)
		for _type in self.data:
			for example in self.data[_type]:
				# populate features
				try:
					example = example["video_identification_features"]
				except KeyError:
					continue
				for flow in example:
					this_ex = example[flow]
					features = self.get_features(this_ex)
					if not features:
						continue
					lab = int(this_ex["is_video"])
					
					self.X["all"].append(features)
					self.Y["all"].append((lab,))
					self.metadata["all"].append((_type,)) # type

		self.X["train"], self.Y["train"], self.X["val"], self.Y["val"] = get_even_train_split(self.X["all"], self.Y["all"], self.train_proportion)

	def predict(self, example):
		# It is assumed a saved model is in the model directory
		# 
		if not self.model:
			model_fn = os.path.join(self.model_dir, self.model_name)
			if not os.path.exists(model_fn):
				raise FileNotFoundError("You need to make a video classifier model.")
			self.model = pickle.load(open(model_fn, 'rb'))

		return self.model.predict(example)

	def save_fig(self, fig_file_name,tight=False):
		# helper function to save to specific figure directory
		if not tight:
			plt.savefig(os.path.join(self.fig_dir, self.figure_prefix + fig_file_name))
		else:
			plt.savefig(os.path.join(self.fig_dir, self.figure_prefix + fig_file_name), 
				bbox_inches='tight', pad_inches=.5)
		plt.clf()
		plt.close()


	def run(self):
		self.load_data()
		self.make_train_val()
		self.train_and_evaluate()

class Video_Classifier_v1:
	# classifies pcaps as containing video or not
	# performs well, because its quite easy to tell by looking at IP's / ASNs
	def __init__(self):
		self.features_dir = "./features"

		self.train_proportion = .5
		self._types = ["twitch", "netflix", "youtube", "no_video"]
		self.type_to_label_mapping = {_type : i for i,_type in enumerate(self._types)}

		self.data = {_type: [] for _type in self._types}

	def train_and_evaluate(self):
		from sklearn.ensemble import RandomForestClassifier
		clf = RandomForestClassifier(max_features="sqrt", n_estimators=3)
		print("Fitting random forest model...")
		clf.fit(self.X["train"], self.Y["train"])
		print("Computing metrics...")
		y_pred = clf.predict(self.X["val"])
		from sklearn.metrics import confusion_matrix, accuracy_score
		conf_mat = confusion_matrix(self.Y["val"], y_pred)
		accuracy = accuracy_score(self.Y["val"], y_pred)
		print("Conf Mat: {} \n\n Accuracy: {}".format(conf_mat, accuracy))


	def make_train_val(self):
		# Creates self.X (train, val) and self.Y (train, val)
		self.X = {"train": [], "val": [], "all": []}
		self.Y = {"train": [], "val": [], "all": []}

		for _type in self.data:
			lab = self.type_to_label_mapping[_type]
			for example in self.data[_type]:
				if isinstance(example["other_statistics"]["ip_likelihood"],float) and _type == "youtube": #  TMP
					example["other_statistics"]["ip_likelihood"] = [0,example["other_statistics"]["ip_likelihood"],0]
				example_features = [el for el in example["other_statistics"]["ip_likelihood"]]
				[example_features.append(el) for el in example["other_statistics"]["asn_dist"]]
				example_features.append(example["other_statistics"]["n_total_asns"])
				self.X["all"].append(example_features)
				self.Y["all"].append(lab)
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
			[self.data[features_type].append(v) for v in features.values()]

	def run(self):
		self.load_data()

		self.make_train_val()

		self.train_and_evaluate()

def main():
	vc = Video_Classifier_v2()
	vc.run()

if __name__ == "__main__":
	main()