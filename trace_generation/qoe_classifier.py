## uses temporal data to try and capture qoe metrics
# Qoe metrics are quality, state, buffer level (high vs low)
import glob, pickle, numpy as np, os, re, time, itertools, traceback

from scipy import stats

import geoip2.database
import matplotlib.pyplot as plt
import tensorflow as tf

from constants import *
from helpers import *
from video_classifier import Video_Classifier_v2


def normalized_acc_metric(y_true, y_pred):
	cf = tf.math.confusion_matrix(tf.math.argmax(y_true,axis=1), tf.math.argmax(y_pred,axis=1))
	c = tf.math.reduce_sum(cf,axis=1)
	expanded_c = tf.tile(c, tf.expand_dims(tf.size(c),axis=0))
	expanded_c = tf.transpose(tf.reshape(expanded_c, (tf.size(c),tf.size(c))))
	normalized_cf = tf.divide(cf, expanded_c)
	normalized_cf = tf.where(tf.math.is_nan(normalized_cf), tf.zeros_like(normalized_cf), normalized_cf)
	normalized_acc = tf.linalg.trace(normalized_cf) / tf.cast(tf.size(c),tf.float64)
	return normalized_acc

# y_true = tf.random.normal((10, 10)) + 30
# y_pred = tf.random.normal((10,10)) + 30
# print(normalized_acc_metric(y_true,y_pred))
# exit(0)



class K2_Model:
	"""VGG-like classifier. Uses temporal features & non-temporal, expert features."""
	def __init__(self, batch_size, input_shape, output_shape):
		self.batch_size = batch_size
		self.input_shape = input_shape
		self.output_shape = output_shape

	def build(self):
		# Define the model here
		inputs = tf.keras.layers.Input(
			shape=self.input_shape, 
			batch_size=self.batch_size
		)
		self.inputs = inputs

		temporal_image = inputs[:,0:-1,:,:]
		byte_stats = temporal_image[:,:,:,0:2]
		dup_acks = temporal_image[:,:,:,2:4]
		cumulative_byte_stats = temporal_image[:,:,:,4:6]
		get_requests = temporal_image[:,:,:,6]
		
		# Temporal image -- leverage spatial correlations
		net1 = byte_stats
		net1 = tf.keras.layers.Conv2D(
			8,
			(N_FLOWS,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net1)
		net1 = tf.keras.layers.Conv2D(
			8,
			(1,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net1)
		net1 = tf.keras.layers.MaxPool2D(pool_size=(1,2),strides=(1,2))(net1)
		net1 = tf.keras.layers.Conv2D(
			16,
			(1,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net1)
		net1 = tf.keras.layers.Conv2D(
			16,
			(1,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net1)
		net1 = tf.keras.layers.MaxPool2D(pool_size=(1,2),strides=(1,2))(net1)
		net1 = tf.keras.layers.Flatten()(net1)


		# Temporal image -- leverage spatial correlations
		net2 = dup_acks
		net2 = tf.keras.layers.Conv2D(
			8,
			(N_FLOWS,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net2)
		net2 = tf.keras.layers.Conv2D(
			8,
			(1,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net2)
		net2 = tf.keras.layers.MaxPool2D(pool_size=(1,2),strides=(1,2))(net2)
		net2 = tf.keras.layers.Conv2D(
			16,
			(1,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net2)
		net2 = tf.keras.layers.Conv2D(
			16,
			(1,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net2)
		net2 = tf.keras.layers.MaxPool2D(pool_size=(1,2),strides=(1,2))(net2)
		net2 = tf.keras.layers.Flatten()(net2)


		# Temporal image -- leverage spatial correlations
		net3 = cumulative_byte_stats
		net3 = tf.keras.layers.Conv2D(
			8,
			(N_FLOWS,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net3)
		net3 = tf.keras.layers.Conv2D(
			8,
			(1,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net3)
		net3 = tf.keras.layers.MaxPool2D(pool_size=(1,2),strides=(1,2))(net3)
		net3 = tf.keras.layers.Conv2D(
			16,
			(1,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net3)
		net3 = tf.keras.layers.Conv2D(
			16,
			(1,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net3)
		net3 = tf.keras.layers.MaxPool2D(pool_size=(1,2),strides=(1,2))(net3)
		net3 = tf.keras.layers.Flatten()(net3)


		net4 = tf.keras.layers.Flatten()(get_requests)
		net4 = tf.keras.layers.Dense(
			20,
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net4)
		net4 = tf.keras.layers.Dense(
			10
		)(net4)

		# Hand-crafted features
		custom_features = inputs[:,-1,:,:]
		net5 = tf.keras.layers.Flatten()(custom_features)
		net5 = tf.keras.layers.Dense(
			20,
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net5)
		net5 = tf.keras.layers.Dense(
			10
		)(net5)

		net = tf.keras.layers.Concatenate()([net1,net2,net3,net4,net5])
		net = tf.keras.layers.Dense(
			4*self.output_shape,
			activation=tf.keras.activations.relu,
		)(net)
		net = tf.keras.layers.Dropout(.15)(net)
		net = tf.keras.layers.Dense(
			2*self.output_shape,
			activation=tf.keras.activations.relu,
		)(net)
		net = tf.keras.layers.Dense(
			self.output_shape,
			activation=tf.keras.activations.softmax,
		)(net)

		self.outputs = net

class K_Model:
	"""VGG-like classifier. Uses temporal features & non-temporal, expert features."""
	def __init__(self, batch_size, input_shape, output_shape):
		self.batch_size = batch_size
		self.input_shape = input_shape
		self.output_shape = output_shape

	def build(self):
		# Define the model here
		inputs = tf.keras.layers.Input(
			shape=self.input_shape, 
			batch_size=self.batch_size
		)
		self.inputs = inputs

		temporal_image = inputs[:,0:-1,:,:]
		
		# Temporal image -- leverage spatial correlations
		net1 = temporal_image
		net1 = tf.keras.layers.Conv2D(
			32,
			(N_FLOWS,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net1)
		net1 = tf.keras.layers.Conv2D(
			32,
			(1,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net1)
		net1 = tf.keras.layers.MaxPool2D(pool_size=(1,2),strides=(1,2))(net1)
		net1 = tf.keras.layers.Conv2D(
			64,
			(1,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net1)
		net1 = tf.keras.layers.Conv2D(
			64,
			(1,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net1)
		net1 = tf.keras.layers.MaxPool2D(pool_size=(1,2),strides=(1,2))(net1)
		

		net1 = tf.keras.layers.Flatten()(net1)


		# Hand-crafted features
		custom_features = inputs[:,-1,:,:]
		net2 = tf.keras.layers.Flatten()(custom_features)
		net2 = tf.keras.layers.Dense(
			20,
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net2)
		net2 = tf.keras.layers.Dense(
			10
		)(net2)

		net = tf.keras.layers.Concatenate()([net1,net2])
		net = tf.keras.layers.Dropout(.5)(net)
		net = tf.keras.layers.Dense(
			2*self.output_shape,
			activation=tf.keras.activations.relu,
		)(net)
		net = tf.keras.layers.Dense(
			self.output_shape,
			activation=tf.keras.activations.softmax,
		)(net)

		self.outputs = net



class QOE_Classifier:
	def __init__(self, skip_load=False, recover_model=False):
		self.skip_load=skip_load
		self.features_dir = "./features"
		self.metadata_dir = METADATA_DIR
		self.fig_dir = "./figures"
		self.figure_prefix = ""
		self.pcap_dir = "./pcaps"
		self.train_proportion = .9
		self.append_tv = True
		# Shape of the input byte statistics image
		self.history_length = HISTORY_LENGTH
		self.n_flows = N_FLOWS
		# self._types = ["twitch", "netflix", "youtube"]
		self._types = ["twitch"]
		self.type_to_label_mapping = {_type : i for i,_type in enumerate(self._types)}
		self.sessions_to_clean = {t:[] for t in self._types}
		# maximum download size -- we divide by this value; could make this depend on the application
		self.max_dl = BYTE_TRANSFER_NORM
		self.ack_norm = DUP_ACK_NORM # normalize number of duplicate acks per T_INTERVAL by this value


		self.X = {"train": {}, "val": {}, "all": []}
		self.Y = {"train": {}, "val": {}, "all": []}
		self.Y_reg = {"train": {}, "val": {}, "all": []}
		self.metadata = {"train": [], "val": [], "all": []} # contains info about each example, in case we want to sort on these values

		self.data = {_type: [] for _type in self._types}
		self.visual_quality_labels = {"high": 2,"medium":1,"low":0}
		
		n_bitrates = 4 #
		self.bitrate_labels = {str(i):i for i in range(n_bitrates)}
		
		self.label_types = ["quality", "buffer", "state", "bitrate", "buffer_delta"]
		self.models = {lt: None for lt in self.label_types}
		self.label_to_label_index = {
			'quality': 0, 
			'buffer': 1, 
			'state': 2, 
			"bitrate": 3, 
			"buffer_delta": 4, 
		}
		self.quality_string_mappings = {
			"twitch": {
				"1600x900": self.visual_quality_labels["high"],
				"1664x936": self.visual_quality_labels["high"],
				"1536x864": self.visual_quality_labels["high"],
				"852x480": self.visual_quality_labels["medium"],
				"1280x720": self.visual_quality_labels["high"],
				"1920x1080": self.visual_quality_labels["high"],
				"1704x960": self.visual_quality_labels["high"],
				"640x360": self.visual_quality_labels["low"],
				"640x480": self.visual_quality_labels["low"],
				"256x144": self.visual_quality_labels["low"],
				"284x160": self.visual_quality_labels["low"],
				"1792x1008": self.visual_quality_labels["high"],
			},
			"netflix": {
				"960x540": self.visual_quality_labels["medium"],
				"960x720": self.visual_quality_labels["medium"],
				"1280x720": self.visual_quality_labels["high"],
				"768x432": self.visual_quality_labels["low"],
				"720x480": self.visual_quality_labels["low"],
				"720x540": self.visual_quality_labels["low"],
				"384x288": self.visual_quality_labels["low"],
				"320x240": self.visual_quality_labels["low"],
				"512x384": self.visual_quality_labels["low"],
			},
		}

		n_buffer_classes = 6
		# Precision at which we'd like to predict buffer health
		precisions = {"twitch": 4, "youtube": 2, "netflix": 5}
		self.buffer_intervals = {
			t: {
				i: [i*precisions[t], (i+1)*precisions[t]] for i in range(n_buffer_classes)
			} for t in precisions
		}
		for t in self.buffer_intervals:
			self.buffer_intervals[t][n_buffer_classes-1][1] = np.inf # no limit on the last class
		self.buffer_health_labels = {i:i for i in range(n_buffer_classes)}

		# Buffer delta intervals / classes
		lowest_vals = { k: -1*n_buffer_classes*precisions[k] / 2 for k in precisions }
		self.buffer_delta_intervals = {
			t: {
				i: [i*precisions[t] + lowest_vals[t], (i+1) * precisions[t] + lowest_vals[t]] for i in range(n_buffer_classes)
			} for t in precisions
		}
		for t in self.buffer_delta_intervals:
			self.buffer_delta_intervals[t][0][0] = -np.inf
			self.buffer_delta_intervals[t][n_buffer_classes-1][1] = np.inf
		self.buffer_delta_labels = {i:i for i in range(n_buffer_classes)}
		self.precisions = precisions

		# maybe in the future there will be a more general way of doing this
		# but bitrates for a service tend to fall into bins
		# get the bins via data exploration
		self.bitrate_intervals = { # kbps
			"twitch": {
				"0": [0,300],
				"1": [300,800],
				"2": [800,1800],
				"3": [1800,1000000]
				#"3": [1800,2500],
				# "4": [2500,6000],
				# "5": [6000,1000000000]
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
				"4": self.state_labels["good"], # paused, but ready to play
				"8": self.state_labels["good"], # playing
				"14": self.state_labels["good"],  # loading data
				"9": self.state_labels["bad"],  # playing/loading
				"5": self.state_labels["bad"], # paused/loadng
				"19": self.state_labels["bad"], # Low buffer / buffering
				"44": self.state_labels["good"], # paused, but can play
				"45": self.state_labels["bad"], # paused, low buffer (so maybe not ready to play?)
				"49": self.state_labels["good"], # Choose a new video
			}
		}
		self.all_labels = {
			"quality": self.visual_quality_labels, 
			"buffer": self.buffer_health_labels, 
			"state": self.state_labels,
			"bitrate": self.bitrate_labels,
			"buffer_delta": self.buffer_delta_labels,
		}
		self.actual_labels = None
		self.vc = Video_Classifier_v2()

		# Model parameters
		self.batch_size = 128
		self.learning_rate = .001
		self.num_epochs = 20
		self.recover_model = recover_model # whether or not to recover when training
		self.total_n_channels = TOTAL_N_CHANNELS
		self.x_inp_shape = (self.n_flows + 1, self.history_length, self.total_n_channels)

		# Data aggregation parameters (see self.temporal_group_session)
		self.frame_aggregation_type = ['mode', 'average'][0]
		self.frame_length = 1 # how many frames over which to aggregate labels

	def cleanup(self):
		""" Log files, pcaps, etc.. may be accidentally deleted over time. Get rid of these in examples, since we are missing information."""
		for _type in self.sessions_to_clean:
			fn = os.path.join(self.features_dir, "{}-features.pkl".format(_type))
			these_features = pickle.load(open(fn, 'rb'))
			for _id in self.sessions_to_clean[_type]:
				print("Deleting ID {}, Type: {}".format(_id,_type))
				del these_features[_id]
			pickle.dump(these_features, open(fn,'wb'))

	def get_augmented_features(self, features):
		""" Perform data augmentation, to increase the number of examples we have. """
		all_permutations = list(itertools.permutations(list(range(self.n_flows))))

		perm = all_permutations[np.random.randint(len(all_permutations))]
		# The order of the rows is meaningless, so we can permute them for more data
		permed_time = features[perm,:,:]
		other_features = np.expand_dims(features[-1,:,:], axis=0)
		permutated_features = np.concatenate([permed_time,other_features],axis=0)
		return permutated_features

	def get_train_iterator(self):
		# Shuffle the data
		perm = list(range(len(self.Y['train'])))
		np.random.shuffle(perm)
		self.X['train'] = self.X['train'][perm]
		self.Y['train'] = self.Y['train'][perm]

		#cat_y_train = list(map(tf.keras.utils.to_categorical,[self.Y["train"]]))[0]
		cat_y_train = self.Y['train']

		for lab, ex in zip(cat_y_train, self.X['train']):
			aug_ex = self.get_augmented_features(ex)
			yield aug_ex, lab

	def get_qoe_label(self, example, i, _type):
		def buffer_health_to_label(buffer_health_seconds):
			for quality in self.buffer_intervals[_type]:
				if buffer_health_seconds >= self.buffer_intervals[_type][quality][0] and\
					buffer_health_seconds < self.buffer_intervals[_type][quality][1]:
					return self.buffer_health_labels[quality]
		def buffer_delta_to_label(buffer_delta_seconds):
			for delta in self.buffer_intervals[_type]:
				if buffer_delta_seconds >= self.buffer_delta_intervals[_type][delta][0] and\
					buffer_delta_seconds < self.buffer_delta_intervals[_type][delta][1]:
						return self.buffer_delta_labels[delta]
		def bitrate_to_label(bitrate_kbps):
			for bitrate in self.bitrate_intervals[_type]:
				if bitrate_kbps >= self.bitrate_intervals[_type][bitrate][0] and\
					bitrate_kbps < self.bitrate_intervals[_type][bitrate][1]:
					return self.bitrate_labels[bitrate]


		# i = 0 -> first report; each additional index corresponds to T_INTERVAL seconds
		t_start = float(example["stats_panel"][0]["timestamp"])
		found=False
		# Finds the first report that occurs after the interval of interest
		for j, report in enumerate(example["stats_panel"]):
			if i * T_INTERVAL <= (float(report["timestamp"]) - t_start):
				found=True
				break
		if not found:
			return None
		report_of_interest = example["stats_panel"][j]
		# no report that corresponds to this time slot
		if np.abs(float(report_of_interest["timestamp"]) - t_start - i * T_INTERVAL) > self.precisions[_type] / 2: 
			return None 

		# Finds the report at the beginning of the sequence (so we can identify changes)
		i_at_beginning = np.maximum(i - T_INTERVAL * self.history_length, 0) # max with 0, in case we are less than history_length in 
		found=False
		# Finds the first report that occurs after the interval of interest
		for j, report in enumerate(example["stats_panel"]):
			if i_at_beginning * T_INTERVAL <= (float(report["timestamp"]) - t_start):
				found=True
				break
		if not found:
			return None
		report_at_beginning = example["stats_panel"][j]
		# no report that corresponds to this time slot
		if np.abs(float(report_at_beginning["timestamp"]) - t_start - i_at_beginning * T_INTERVAL) > self.precisions[_type] / 2: 
			return None 
		

		state = report_of_interest["state"].strip().lower()
		quality = report_of_interest["current_optimal_res"] # visual quality
		buffer_health = float(report_of_interest["buffer_health"].replace("s",""))
		buffer_health_beginning = float(report_at_beginning["buffer_health"].replace("s",""))
		bitrate = float(report_of_interest["bitrate"]) # kbps

		# Convert the raw buffer health to an integer label
		buffer_health_label = buffer_health_to_label(buffer_health)
		buffer_delta_label = buffer_delta_to_label(buffer_health - buffer_health_beginning)
		bitrate_label = bitrate_to_label(bitrate)

		# if np.random.random() > .9:
		# 	print("Report: {}, report beginning: {}, bh: {}, bhd: {}".format(report_of_interest, report_at_beginning, buffer_health_label, buffer_delta_label))
		# 	if np.random.random() > .99:
		# 		exit(0)
		if _type == "twitch":
			if quality == "0x0":
				return None
			quality_label = self.quality_string_mappings[_type][quality]
		elif _type == "netflix":
			quality_label = self.quality_string_mappings[_type][quality]
		elif _type == "youtube":
			# youtube tells you what the maximum quality is, so we treat it differently
			if quality == "0x0":
				return None
			try:
				experienced,optimal = quality.split("/")
			except:
				print(quality);exit(0)
			experienced = [int(el) for el in experienced.split("@")[0].split("x")][1]
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

		try:
			state_label = self.video_states[_type][state]
		except KeyError:
			print(report_of_interest)
			print("Note -- state {} for type {} not found, returning.".format(state, _type))
			return None

		# Some types of labels don't make sense
		# If video is paused or buffered, quality is undefined
		if state_label == self.state_labels['bad']:
			# TODO -- set this to None or something like that
			quality_label = self.visual_quality_labels['low']
			bitrate_label = self.bitrate_labels['0']
			buffer_health_label = self.buffer_health_labels[0]
		if buffer_health_label == self.buffer_health_labels[0]:
			# likely rebuffering
			state_label = self.state_labels["bad"]

		self.actual_labels = [quality, buffer_health, state, bitrate]
		class_label = [quality_label, buffer_health_label, state_label,\
		 	bitrate_label, buffer_delta_label]
		reg_label = [buffer_health]
		
		return class_label, reg_label

	def get_model_name(self, label_type):
		# construct a name for a model, depending on various hyper-parameters
		if label_type == "buffer":
			params_to_name_from = [self.history_length, self.n_flows, self.frame_length]
		elif label_type == "buffer_delta":
			params_to_name_from = [self.history_length, self.n_flows, self.frame_length]
		else:
			raise ValueError("Getting model name for problem type {} not yet implemented.".format(label_type))
		return "-".join([str(el) for el in params_to_name_from])

	def get_recent_frames(self, i, arr):
		# arr is an array of temporal values, self.n_flows x total_length x 2
		# extract the most recent values, and put them on the right side of the image

		# HAVE TO BE CAREFUL BECAUSE SLICING IS ASSIGNING BY REFERENCE
		tmp = arr[:,0:i,:]

		if i < self.history_length:
			# Fill in non-existent past with zeros
			tmp2 = np.concatenate([
				np.zeros((self.n_flows, self.history_length - i, 2)),
				tmp], axis=1)
		else:
			tmp2 = tmp[:,-self.history_length:,:]

		return tmp2

	def train_and_evaluate(self, label_type):
		n_classes = len(self.all_labels[label_type])
		# Need to make data set lengths a perfect multiple of the number of batches, 
		# or 'fit' fails
		n_feed_ins = {}
		for t in ['train', 'val']:
			n_feed_ins[t] = int(len(self.X[t]) / self.batch_size)
			self.X[t] = self.X[t][0:n_feed_ins[t]*self.batch_size]
			self.Y[t] = self.Y[t][0:n_feed_ins[t]*self.batch_size]
		#cat_y_val = list(map(tf.keras.utils.to_categorical,[self.Y["val"]]))[0]
		cat_y_val = self.Y["val"]
		# Get the training data generator obj
		train_data = tf.data.Dataset.from_generator(self.get_train_iterator, 
			output_types=(tf.float32, tf.float32),
			#output_shapes=((self.n_flows + 1,self.history_length,2), (n_classes)))
			output_shapes=(self.X['train'][0].shape, (n_classes)))
		train_data = train_data.repeat().batch(self.batch_size)
		# Get the model object
		model = K2_Model(self.batch_size, self.x_inp_shape, n_classes)
		model.build()
		keras_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.outputs)
		# Recovering and saving models
		model_name = self.get_model_name(label_type)
		chkpt_path = os.path.join(MODEL_DIR, label_type, model_name)
		if self.recover_model:
			if not os.path.exists(chkpt_path):
				raise ValueError('Model does not have any checkpoints saved to initialize from.  Set recover_model to False.')
			else:
				print("Loading model from {}".format(chkpt_path))
			keras_model.load_weights(os.path.join(chkpt_path, model_name))
		else:
			clear_path(chkpt_path)
			make_path(chkpt_path)
		
		# for saving models at each epoch
		cp_callback = tf.keras.callbacks.ModelCheckpoint(
			filepath=os.path.join(chkpt_path, model_name),
			save_weights_only=True,
			verbose=1
		)

		# Set up training and validation
		keras_model.compile(
			optimizer=tf.keras.optimizers.Adam(
				learning_rate=self.learning_rate
			), 
			loss='categorical_crossentropy', 
			metrics=['accuracy', normalized_acc_metric])
		keras_model.fit(train_data,
			batch_size=self.batch_size,
			validation_data=(self.X["val"], cat_y_val), 
			callbacks=[cp_callback],
			steps_per_epoch=n_feed_ins['train'],
			epochs=self.num_epochs,
			verbose=2)
		preds = keras_model.predict(self.X["val"])
		preds = np.argmax(preds, axis=1)
		self.Y["val"] = [np.argmax(el) for el in self.Y["val"]]
		cf = tf.math.confusion_matrix(self.Y["val"], preds, n_classes)
		normalized_cf = cf/np.transpose(np.tile(np.sum(cf,axis=1), (n_classes,1)))
		normalized_acc = 1 -sum(normalized_cf[i,j]/cf.shape[0] for i in range(cf.shape[0]) for j in range(cf.shape[1])if i != j)
		print("Normalized CF: {}\n Nomalized Acc: {}".format(normalized_cf, normalized_acc))
		
		n_off = np.zeros((len(normalized_cf),))
		for i in range(len(normalized_cf)):
			for j in range(len(normalized_cf)):
				n_off[np.abs(i-j)] += normalized_cf[i,j] / len(normalized_cf)
		for i in range(len(n_off) - 1):
			print("Acc counting {} off: {} ".format(i, sum(n_off[0:i+1])))

		# Tabulate specifically, what it is getting wrong
		quals, types = {}, {}
		uid = {}
		n_val = len(self.Y['val'])
		n_to_plot = 10
		incorrects = []
		for example, pred, lab, md in zip(self.X['val'],preds, self.Y['val'], self.metadata['val']):
			if pred != lab:
				# Select most common qual
				all_quals = [el["actual_labels"][0] for el in md]
				qual = stats.mode(all_quals)[0][0]
				if qual == '284x160':
					incorrects.append((example, [pred,None,None], [qual, None, None]))
				if len(incorrects) == n_to_plot:
					self.visualize_example(np.random.random(), incorrects)
				_type = md[0]["type"]
				try:
					quals[qual] += 1.0/n_val
					uid[qual,_type] += 1.0/n_val
				except KeyError:
					quals[qual] = 1.0/n_val
					uid[qual,_type] = 1.0/n_val
				try:
					types[_type] += 1.0/n_val
				except KeyError:
					types[_type] = 1.0/n_val
		# print(uid)
		# print(quals)
		# print(types)
		exit(0)
		# Visualize incorrect preds
		n_to_plot = 10
		incorrects = []
		for pred, lab, example in zip(preds, self.Y['val'], self.X['val']):
			if pred != lab:
				incorrects.append((example,[pred,None,None],[lab,None,None]))
			if len(incorrects) == n_to_plot:
				self.visualize_example(np.random.random(), incorrects)
				incorrects = []

	def load_model(self, label_type):
		"""Loads a pre-trained model."""
		model_name = self.get_model_name(label_type)
		n_classes = len(self.all_labels[label_type])
		model = K2_Model(self.batch_size, self.x_inp_shape, n_classes)
		model.build()
		keras_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.outputs)
		# Recovering and saving models
		model_name = self.get_model_name(label_type)
		chkpt_path = os.path.join(MODEL_DIR, label_type, model_name)
		try:
			keras_model.load_weights(os.path.join(chkpt_path, model_name))
		except:
			print(traceback.print_exc())
			raise ValueError("No model exists in {}".format(chkpt_path))
		self.models[label_type] = keras_model

	def make_train_val(self):
		# Creates self.X (train, val) and self.Y (train, val)
		print("Making train-val.")
		for _type in self.data:
			for example in self.data[_type]:
				
				# Load byte statistics, and create a set of features of constant shape
				# we can use as input to the model
				byte_statistics = example["byte_statistics"]

				# Group features and labels by session
				session_features, session_labels, session_metadata = [], [], []

				# Limit to only video flows
				all_feats = []
				for flow in byte_statistics[0].keys():
					video_ex = example["video_identification_features"][flow]
					all_feats.append((flow, self.vc.get_features(video_ex)))
				all_feats = [el for el in all_feats if el[1]]
				all_labs = self.vc.predict([el[1] for el in all_feats])
				
				# This is non-causal as coded, but a causal version 
				# would provide an equivalent result
				# The actual implementation of this process would be so different, it
				# doesn't really matter

				all_flows = [flow for lab, (flow, feat) in zip(all_labs, all_feats) if lab]
				all_non_video_flows = [flow for lab, (flow, feat) in zip(all_labs, all_feats) if not lab]


				# Vote on the type 
				pred_service_type = stats.mode([lab for lab in all_labs if lab])[0]

				# Time into the packet captures that we start recording 
				# video playback statistics
				t_start_recording_offset = example["start_offset"]
				bin_start = int(np.floor(t_start_recording_offset / T_INTERVAL))
				n_bins = len(byte_statistics[0][list(all_flows)[0]])
				byte_stats = np.zeros((self.n_flows, n_bins, 2)) 
				# dict with self.n_flows keys; each key is index of flow in 
				# all_flows -> row in byte_stats this flow occupies
				current_best_n = {} 

				# If we have fewer than self.n_flows video flows in total (over the whole session), 
				# Append some dummy (zero) information
				if len(all_flows) < self.n_flows:
					n_to_add = self.n_flows - len(all_flows)
					for i in range(n_to_add):
						byte_statistics[0][i,i] = np.zeros((n_bins))
						byte_statistics[1][i,i] = np.zeros((n_bins))
						all_flows.append((i,i))

				active_flows, non_video_total_bytes = {}, {}
				# Flow acks contains all ack #s (and times) for a given flow (u/d)
				# dup acks maps time instances to binary arr of length 2xself.n_flows
				# Indicating if any of the flows in the array experienced a duplicate ACK (loss)
				# the '2' is for u/d
				flow_acks = {flow: [{}, {}] for flow in all_flows}
				dup_acks_by_t = {i: [[], []] for i in range(n_bins)} # list of dup acks by time bin
				dup_acks = np.zeros((self.n_flows,n_bins,2))
				for i in range(n_bins):
					sum_flows = []
					for flow in all_flows:
						if i < self.history_length:
							s = 0
						else:
							s = i - self.history_length + 1
						up_data = np.sum(byte_statistics[0][flow][s:i+1])
						down_data = np.sum(byte_statistics[1][flow][s:i+1])
						sum_flows.append(up_data + down_data)
						if flow[0] == flow[1]: # dummy filler flow:
							continue
						for j in range(2): # u/d
							for transp_data in byte_statistics[2+j][flow][i]:
								try:
									seq_n, ack_n = transp_data
								except ValueError:
									seq_n, ack_n, _ = transp_data
								try:
									flow_acks[flow][j][ack_n].append(i)
									# duplicate ack
									#print("Dup ACK!")
									try:
										dup_acks_by_t[i][j].append(flow)
									except KeyError:
										dup_acks_by_t[i][j] = [flow]
								except KeyError:
									flow_acks[flow][j][ack_n] = [i]


					# Accumualate non-video bytes (expert feature)
					non_video_total_bytes[i] = 0
					for flow in all_non_video_flows:
						non_video_total_bytes[i] += np.sum(byte_statistics[0][flow][0:i+1])
						non_video_total_bytes[i] += np.sum(byte_statistics[1][flow][0:i+1])
					# Record which flows are active for later
					active_flows[i] = {flow: sum_flows[j] > 0 for j, flow in enumerate(all_flows)}
					# Indices of highest data transferred N flows in 
					# all_flows
					best_n = np.argpartition(sum_flows,-1*self.n_flows)[-1*self.n_flows:]
					if current_best_n == {}:
						current_best_n = {best: i for i,best in enumerate(best_n)}
					if set(best_n) != set(current_best_n.keys()):
						new_best_flows = get_difference(best_n, current_best_n.keys())
						flows_to_remove = get_difference(current_best_n.keys(), best_n)
						for add_flow, remove_flow in zip(new_best_flows, flows_to_remove):
							i_of_new_flow = current_best_n[remove_flow]
							del current_best_n[remove_flow]
							current_best_n[add_flow] = i_of_new_flow
					#print("i: {}, SF: {}, best n: {}, {}".format(i, sum_flows, best_n, [sum_flows[j] for j in best_n]))
					for k in best_n:
						for j in [0,1]:
							# all_flows[k] is the flow in question
							# current_best_n[k] is the index corresponding index in byte stats
							# Populate byte stats with most active flows
							byte_stats[current_best_n[k]][i][j] = byte_statistics[j][all_flows[k]][i]
							# Populate corresponding expert features
							dup_acks[current_best_n[k],i,j] = dup_acks_by_t[i][j].count(all_flows[k])

				size_byte_stats = byte_stats.shape
				to_viz, n_to_plot = [], 40
				total_bytes = np.cumsum(byte_stats,axis=1)
				for i in range(bin_start, size_byte_stats[1] - self.history_length):
					# Get flows active up to this point
					active_flows_now = active_flows[i]
					if sum(active_flows_now.values()) == 0:
						continue

					# Form hand-crafted features
					# Total non-video bytes up to this point
					non_video_bytes = non_video_total_bytes[i] / (10 * self.max_dl)
					
					other_temporal_features = np.zeros((self.n_flows, self.history_length, self.total_n_channels - 2))
					# Total bytes for each (video) flow (u/d)
					total_bytes_ud = self.get_recent_frames(i, total_bytes) / ( self.max_dl * 10 ) # want to make it roughly the same scale as the other features 
					# dup ack instaces 
					dup_acks_ud = self.get_recent_frames(i, dup_acks) / self.ack_norm
					other_temporal_features[:,:,0:2] = dup_acks_ud
					other_temporal_features[:,:,2:4] = total_bytes_ud


					
					expert_features = np.zeros((1,self.history_length,self.total_n_channels))
					expert_features[0,0,0] = i / MAX_TIME # non_video_bytes
					expert_features[0,1,0] = pred_service_type
					# Other ideas -- elapsed time

					# End handcrafted features



					# Form temporal image, showing byte transfers
					# only reveal up to the current timestep

					temporal_image = self.get_recent_frames(i, byte_stats)
					get_request_indicator = (temporal_image[:,:,0] > GET_REQUEST_SIZE).astype(np.int64)
					other_temporal_features[:,:,4] = get_request_indicator


					# Clip between 0 and 1 (not guaranteed, but likely)
					temporal_image = temporal_image / self.max_dl

					# label for this time step
					labs = self.get_qoe_label(example,i - bin_start, _type)
					
					if labs is None:
						continue
					actual_labels = self.actual_labels
					class_lab, reg_lab = labs

					if False:
						to_viz.append((temporal_image,class_lab,actual_labels))
						if len(to_viz) == n_to_plot:
							self.visualize_example("{}-{}".format(_type, example['_id']), 
								to_viz)
							exit(0)

					# Append all the different types of features into one array
					features = np.concatenate([temporal_image, other_temporal_features], axis=2)
					features = np.concatenate([features, expert_features], axis=0)
					session_features.append(features)
					session_labels.append(class_lab)
					session_metadata.append({
						"time_slot": i, # time slot in the capture
						"type": _type, # type of servie
						"actual_labels": actual_labels # un-discretized labels
					})
				
				for feature, lab, metadata in self.temporal_group_session(session_features, 
					session_labels, session_metadata):
					self.X["all"].append(feature)
					self.Y["all"].append(lab)
					self.metadata["all"].append(metadata)
					# TODO - maybe fix
					self.Y_reg["all"].append(reg_lab)

		self.X["train"], self.Y["train"], self.X["val"], self.Y["val"], self.metadata['train'], self.metadata['val'] = get_even_train_split(
			self.X["all"], self.Y["all"], self.metadata['all'], self.train_proportion, is_dist=True)
		# Change the keys from integers to strings for readability
		for tv in ["train", "val"]:
			for i, _problem_type in enumerate(self.label_types):
				self.X[tv][_problem_type] = self.X[tv][i]
				self.Y[tv][_problem_type] = self.Y[tv][i]
				self.metadata[tv][_problem_type] = self.metadata[tv][i]
				del self.X[tv][i]
				del self.Y[tv][i]
				del self.metadata[tv][i]

	def predict_from_model(self, label_type, service_type, _x):

		# Classifier classifies an interval of buffer
		# lower is a cautious classification while mean is possibly more accurate
		method = "lower" # "mean"
		

		if self.models[label_type] is None:
			self.load_model(label_type)
		predicted_distributions = self.models[label_type].predict(_x)
		predicted_classes = [np.argmax(prediction) for prediction in predicted_distributions]
		# Convert predicted classes into the values they represent
		predicted_values = []
		if label_type == "buffer":
			for predicted_class in predicted_classes:
				if method == "mean":
					# Return mean of the buffer interval
					if self.buffer_intervals[service_type][predicted_class][-1] == np.inf:
						predicted_values.append(self.buffer_intervals[service_type][predicted_class][0])
					else:
						predicted_values.append(np.mean(self.buffer_intervals[service_type][predicted_class]))
				else:
					predicted_values.append(self.buffer_intervals[service_type][predicted_class][0])
		else:
			raise ValueError("Label type {} not yet implemented in QOE_Classifier.predict_from_model.".format(label_type))

		return predicted_values

	def save_train_val(self):
		# saves the training and validation sets to pkls
		# each label_type type gets its own train + val set, since each will get its own classifier (at least for now)
		if self.append_tv:
			for label_type in self.label_types:
				# load old data and append new data to it
				t_fn, v_fn = os.path.join(self.features_dir, "{}-{}.pkl".format(label_type,"train")),\
					os.path.join(self.features_dir,"{}-{}.pkl".format(label_type,"val"))
				if not os.path.exists(t_fn) or not os.path.exists(v_fn): continue
				t, v = pickle.load(open(t_fn,'rb')), pickle.load(open(v_fn,'rb'))
				for t_x, t_y, t_m in zip(t['X'], t['Y'], t['metadata']):
					self.X['train'][label_type].append(t_x)
					self.Y['train'][label_type].append(t_y)
					self.metadata['train'][label_type].append(t_m)
				for v_x, v_y, v_m in zip(v['X'], v['Y'], v['metadata']):
					self.X['val'][label_type].append(v_x)
					self.Y['val'][label_type].append(v_y)
					self.metadata['val'][label_type].append(v_m)


		for label_type in self.label_types:
			train = {'X': self.X['train'][label_type], 'Y':self.Y['train'][label_type], 'metadata': self.metadata['train'][label_type]}
			val = {'X': self.X['val'][label_type], 'Y':self.Y['val'][label_type], 'metadata': self.metadata['val'][label_type]}
			t_fn, v_fn = os.path.join(self.features_dir, "{}-{}.pkl".format(
				label_type,"train")), os.path.join(self.features_dir,"{}-{}.pkl".format(
				label_type,"val"))
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
			self.X["train"], self.Y["train"], self.metadata['train'] = np.array(t['X']), np.array(t["Y"]), t['metadata']
			self.X["val"], self.Y["val"], self.metadata['val'] = np.array(v['X']), np.array(v["Y"]), v['metadata']
			# Shuffle validation
			perm = list(range(len(self.X['val'])))
			perm = np.random.permutation(perm)
			self.X['val'] = self.X['val'][perm]
			self.Y['val'] = self.Y['val'][perm]
			self.metadata['val'] = [self.metadata['val'][i] for i in perm]
		else:
			raise ValueError("dtype {} not understood".format(dtype))

	def temporal_group_session(self, features, labels, metadatas):
		"""Takes session features, labels, and metadata (sorted in time), 
			divides them into frames over which we look at average QoE."""
		# THIS IS BROKEN -- I NEED TO EXCLUDE CASES FOR WHICH I DON'T HAVE ENOUGH 
		# DATA FROM CONSECUTIVE FRAMES
		n_frames = int(np.ceil(len(features) / self.frame_length))
		n_labels = len(self.label_types)
		feature_frame, label_frame, metadata_frame = None, np.zeros((self.frame_length, n_labels)), []
		i = 0
		n_classes_per_subproblem = [len(self.all_labels[t]) for t in self.label_types]
		for feature, label, metadata in zip(features, labels, metadatas):
			label_frame[i % self.frame_length,:] = label
			feature_frame = feature
			metadata_frame.append(metadata) # save all metadata
			if i % self.frame_length == self.frame_length - 1:
				aggregated_label = [np.zeros((el)) for el in n_classes_per_subproblem]
				if self.frame_aggregation_type == 'average':
					for j in range(len(n_classes_per_subproblem)):
						# Fill in each sub-problem with its distribution of labels
						u, c = np.unique(label_frame[:,j], return_counts=True)
						u = u.astype(np.int32)
						aggregated_label[j][u] = c / self.frame_length
				elif self.frame_aggregation_type == 'mode':
					for j in range(len(n_classes_per_subproblem)):
						most_common = int(stats.mode(label_frame[:,j])[0][0])
						aggregated_label[j][most_common] = 1.0
				else:
					raise ValueError("Frame agg. type {} not supported.".format(
						self.frame_aggregation_type)) 
				# For now, yield most recent features
				yield feature_frame, aggregated_label, metadata_frame
				feature_frame, label_frame, metadata_frame = None, np.zeros((self.frame_length, n_labels)), []

			i += 1

	def visualize_example(self, _id, examples):
		# Show bytes / time, along with label
		base = 2	
		ar = self.history_length / self.n_flows
		plt.rcParams.update({"figure.figsize": [base*ar,base*int(1.5*len(examples))]})
		fig, ax = plt.subplots(len(examples),1)
		i=0
		for temporal_image, label, actual_labels in examples:
			im = ax[i].imshow(np.sum(temporal_image[0:-1,:,:], axis=2), vmin=0, vmax=1)
			ax[i].set_title("Quality: {} ({}), Buffer: {} ({}), State: {} ({})".format(
				label[0],actual_labels[0],label[1],actual_labels[1], label[2],actual_labels[2]))
			ax[i].set_xlabel("Time")
			ax[i].set_ylabel("Flow Index")
			i += 1
		fig.subplots_adjust(right=.8)
		cbar_ax = fig.add_axes([.85,.15,.05,.07])
		fig.colorbar(im, cax=cbar_ax)

		
		self.save_fig("ex_visualization_{}.pdf".format(_id))
		if np.random.random() > .7:
			exit(0)

	def save_fig(self, fig_file_name,tight=False):
		# helper function to save to specific figure directory
		if not tight:
			plt.savefig(os.path.join(self.fig_dir, self.figure_prefix + fig_file_name))
		else:
			plt.savefig(os.path.join(self.fig_dir, self.figure_prefix + fig_file_name), 
				bbox_inches='tight', pad_inches=.5)
		plt.clf()
		plt.close()

	def run(self, label_type):
		if self.skip_load == "no" or not os.path.exists(os.path.join(self.features_dir, "{}-train.pkl".format(label_type))):
			# only do this if we need to
			self.load_data('raw')
			self.make_train_val()
			self.save_train_val()
		self.load_data('formatted', label_type)
		self.train_and_evaluate(label_type)

		self.cleanup()

def main():
	#  Loads data & trains video QoE classifier of a certain type
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--sl', action='store',default="no")
	parser.add_argument('--pt', action='store', default='quality')
	parser.add_argument('--recover', action='store_true')
	args = parser.parse_args()

	qoec = QOE_Classifier(skip_load=args.sl,recover_model=args.recover)
	problem_type = args.pt
	print("Trying problem type: {}".format(problem_type))
	qoec.run(problem_type)


if __name__ == "__main__":
	main()