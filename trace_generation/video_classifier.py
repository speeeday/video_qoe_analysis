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
		self.metadata_dir = METADATA_DIR
		self.train_proportion = .8
		self.history_length = 100
		#self._types = ["twitch", "netflix", "youtube"]
		self._types = ["netflix"]
		self.type_to_label_mapping = {_type : i for i,_type in enumerate(self._types)}
		self.max_tpt = float(15e6) # maximum download size -- we divide by this value; could make this depend on the application

		self.X = {"train": {}, "val": {}, "all": []}
		self.Y = {"train": {}, "val": {}, "all": []}
		self.metadata = {"train": [], "val": [], "all": []} # contains info about each example, in case we want to sort on these values

		self.data = {_type: [] for _type in self._types}
		self.all_labels = [0,1]

		# Feature processing
		self.no_feature = -1
		self.tpt_log_base = 4
		self.bytes_log_base = 4
		
	def train_and_evaluate(self):
		from sklearn.ensemble import RandomForestClassifier
		clf = RandomForestClassifier(max_features="sqrt", n_estimators=10)
		print("Fitting random forest model...")
		clf.fit(self.X["train"], self.Y["train"])
		print("Computing metrics...")
		y_pred = clf.predict(self.X["val"])
		from sklearn.metrics import confusion_matrix, accuracy_score
		conf_mat = confusion_matrix(self.Y["val"], y_pred)
		accuracy = accuracy_score(self.Y["val"], y_pred)
		print("Conf Mat: {} \n\n Accuracy: {}".format(conf_mat, accuracy))

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

	def visualize_data(self):
		cs = {0:'r',1:'b'}
		x,y = [el[2] for el in self.X['train']], [el[1] for el in self.X['train']]
		x = np.array(x,dtype=np.float64) + .5*np.random.randn(len(x))
		y = np.array(y,dtype=np.float64) + .5*np.random.randn(len(y))
		plt.scatter(x,y, c=[cs[lab] for lab in self.Y['train']])
		plt.xlabel("Bytes Down")
		plt.ylabel("Bytes Up")
		self.save_fig("feature_viz.pdf")

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
					# For all byte-type features, take log and integerize
					# Total Bytes
					if this_ex["total_bytes"] == 0:
						continue
					total_bytes = int(math.log(this_ex["total_bytes"],self.bytes_log_base))
					# Up
					if this_ex["total_bytes_up"] > 0:
						tb_up = int(math.log(this_ex["total_bytes_up"],self.bytes_log_base))
					else:
						tb_up = self.no_feature
					# Down
					if this_ex["total_bytes_down"] > 0:
						tb_down = int(math.log(this_ex["total_bytes_down"],self.bytes_log_base))
					else:
						tb_down = self.no_feature
					# Take log2 of the throughput variables, since finer granularity provides more information
					try:
						mean_tpt = int(math.log(this_ex["mean_throughput"], self.tpt_log_base))
					except KeyError:
						mean_tpt = self.no_feature
					try:
						max_tpt = int(math.log(this_ex["max_throughput"], self.tpt_log_base))
					except KeyError:
						max_tpt = self.no_feature
					try:
						if this_ex["std_throughput"] > 0:
							std_tpt = int(math.log(this_ex["std_throughput"], self.tpt_log_base))
						else:
							std_tpt = self.no_feature
					except KeyError:
						std_tpt = self.no_feature

					lab = int(this_ex["is_video"])
					features = [total_bytes, tb_up, tb_down, mean_tpt, max_tpt, std_tpt]
					# if lab:
					# 	print("Had video: {}".format(features))
					# else:
					# 	print("Didn't have video: {}".format(features))
					# if np.random.random() > .99:
					# 	exit(0)
					self.X["all"].append(features)
					self.Y["all"].append(lab)
					self.metadata["all"].append((_type,)) # type

		n_total_examples = len(self.X["all"])
		limiting_factors = np.zeros(len(self.all_labels))
		for _type in self._types:
			inds = [i for i,el in enumerate(self.metadata["all"]) if el[0] == _type]
			labs = [self.Y["all"][i] for i in inds]
			print("Type: {}, {} total examples".format(_type,len(inds)))
			x,c = np.unique(labs,return_counts=True)
			for el_x,el_c in zip(x,c):
				limiting_factors[el_x] += el_c
			print("{} with counts {}".format(x,c))
		
		# minimum counts of label-- our data set is constrained by this factor
		limiting_factor = np.min(limiting_factors) 
		print("Limiting amount of data is {} examples.".format(limiting_factor))
		from sklearn.model_selection import train_test_split
		examples_by_label = [[x for x,_y in zip(self.X["all"], self.Y["all"]) if _y == y] for y in range(len(self.all_labels))]
		n_train = int(limiting_factor*self.train_proportion)
		n_val = int(limiting_factor - n_train)
		train_example_indices = [np.random.choice(range(len(el)), size=n_train, replace=False) for el in examples_by_label]
		val_example_indices = [get_difference(range(len(el)), tei) for el,tei in zip(examples_by_label, train_example_indices)]
		val_example_indices = [np.random.choice(vei, size=n_val, replace=False) for vei in val_example_indices]

		train_examples_to_save = [[self.X["all"][i] for i in tei] for tei in train_example_indices]
		self.X["train"] = [el for l in train_examples_to_save for el in l]
		self.Y["train"] = [i for i in range(len(train_examples_to_save)) for l in train_examples_to_save[i]]
		
		val_examples_to_save = [[self.X["all"][i] for i in vei] for vei in val_example_indices]
		self.X["val"] = [el for l in val_examples_to_save for el in l]
		self.Y["val"] = [i for i in range(len(val_examples_to_save)) for l in val_examples_to_save[i]]

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
		self.visualize_data()
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