## uses high level features and a random forest classifier to distinguish video from non-video flows
import glob, pickle, numpy as np, os, re

class Simple_Video_Classifier:
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
	svc = Simple_Video_Classifier()
	svc.run()

if __name__ == "__main__":
	main()