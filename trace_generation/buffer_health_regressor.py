## uses temporal data to try and capture qoe metrics
# Qoe metrics are quality, state, buffer level (high vs low)
import glob, pickle, numpy as np, os, re, time, itertools, traceback

from scipy import stats

import geoip2.database
import matplotlib.pyplot as plt
import tensorflow as tf

from constants import *
from helpers import *

np.set_printoptions(threshold=np.inf,precision=3, suppress=True)

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

		net = self.inputs[:,:-1,:,:]

		# temporal_image = inputs[:,0:-1,:,:]
		# byte_stats = temporal_image[:,:,:,0:2]
		# dup_acks = temporal_image[:,:,:,2:4]
		# cumulative_byte_stats = temporal_image[:,:,:,4:6]
		# get_requests = temporal_image[:,:,:,6]
		
		net = tf.keras.layers.Conv2D(
			8,
			(N_FLOWS,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net)
		net = tf.squeeze(net)
		net = tf.expand_dims(net,axis=3)
		net = tf.keras.layers.Conv2D(
			16,
			(3,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net)
		net = tf.keras.layers.Conv2D(
			16,
			(3,3),
			strides=(1,1),
			activation=tf.keras.activations.relu,
			use_bias=True,
		)(net)

		net = tf.keras.layers.Flatten()(net)

		net = tf.keras.layers.Dense(
			20,
			activation=tf.keras.activations.relu,
		)(net)
		net = tf.keras.layers.Dropout(.5)(net)
		net = tf.keras.layers.Dense(
			10,
			activation=tf.keras.activations.relu,
		)(net)
		net = tf.keras.layers.Dense(
			self.output_shape,
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



class Buffer_Regressor:
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
		self.label_norm = 1 # normalize buffer health by this amount (to stabilize training)


		self.X = {"train": {}, "val": {}, "all": []}
		self.Y = {"train": {}, "val": {}, "all": []}
		self.metadata = {"train": {}, "val": {}, "all": []} # contains info about each example, in case we want to sort on these values

		self.data = {_type: [] for _type in self._types}
		
		self.label_types = ["buffer_regression"]
		self.models = {lt: None for lt in self.label_types}

		self.actual_labels = None

		# Model parameters
		self.batch_size = 128
		self.learning_rate = .001
		self.num_epochs = 30
		self.recover_model = recover_model # whether or not to recover when training
		self.total_n_channels = TOTAL_N_CHANNELS
		self.x_inp_shape = (self.n_flows + 1, self.history_length, self.total_n_channels)

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


		for lab, ex in zip(self.Y['train'], self.X['train']):
			aug_ex = self.get_augmented_features(ex)
			yield aug_ex, lab

	def get_qoe_label(self, example, i, _type):
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
		if np.abs(float(report_of_interest["timestamp"]) - t_start - i * T_INTERVAL) > 1.0 / 2: 
			return None 
		

		buffer_health = float(report_of_interest["buffer_health"].replace("s",""))

		self.actual_labels = [buffer_health]
		return [buffer_health / self.label_norm]

	def get_model_name(self, label_type):
		# construct a name for a model, depending on various hyper-parameters
		if label_type == "buffer_regression":
			params_to_name_from = [self.history_length, self.n_flows]
		else:
			raise ValueError("Getting model name for problem type {} not yet implemented.".format(label_type))
		return "-".join([str(el) for el in params_to_name_from])

	def get_recent_frames(self, i, arr):
		# arr is an array of temporal values, self.n_flows x total_length x n_channels
		# extract the most recent values, and put them on the right side of the image

		# HAVE TO BE CAREFUL BECAUSE SLICING IS ASSIGNING BY REFERENCE
		tmp = arr[:,0:i,:]
		n_chans = arr.shape[2] # number of channels in the image

		if i < self.history_length:
			# Fill in non-existent past with zeros
			tmp2 = np.concatenate([
				np.zeros((self.n_flows, self.history_length - i, n_chans)),
				tmp], axis=1)
		else:
			tmp2 = tmp[:,-self.history_length:,:]

		return tmp2

	def train_and_evaluate(self, label_type):
		# Need to make data set lengths a perfect multiple of the number of batches, 
		# or 'fit' fails
		n_feed_ins = {}
		for t in ['train', 'val']:
			n_feed_ins[t] = int(len(self.X[t]) / self.batch_size)
			self.X[t] = self.X[t][0:n_feed_ins[t]*self.batch_size]
			self.Y[t] = self.Y[t][0:n_feed_ins[t]*self.batch_size]
		# Get the training data generator obj
		train_data = tf.data.Dataset.from_generator(self.get_train_iterator, 
			output_types=(tf.float32, tf.float32),
			#output_shapes=((self.n_flows + 1,self.history_length,2), (n_classes)))
			output_shapes=(self.X['train'][0].shape, 1))
		train_data = train_data.repeat().batch(self.batch_size)
		# Get the model object
		model = K2_Model(self.batch_size, self.x_inp_shape, 1)
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
			loss='mean_squared_error')
		keras_model.fit(train_data,
			batch_size=self.batch_size,
			validation_data=(self.X["val"], self.Y["val"]), 
			callbacks=[cp_callback],
			steps_per_epoch=n_feed_ins['train'],
			epochs=self.num_epochs,
			verbose=2)
		preds = keras_model.predict(self.X["val"])
		all_errors = preds - self.Y["val"]
		print("Mean absolute error on validation set: {}".format(np.mean(np.abs(all_errors))))
		# for _x, _y in zip(self.X['val'],preds):
		# 	print("\n\n")
		# 	print(_x)
		# 	print(_y)
		# 	if np.random.random() > .9:
		# 		exit(0)


	def load_model(self, label_type):
		"""Loads a pre-trained model."""
		model_name = self.get_model_name(label_type)
		model = K2_Model(self.batch_size, self.x_inp_shape, 1)
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
		corr_check = []
		for _type in self.data:
			for example in self.data[_type]:
				
				# Load byte statistics, and create a set of features of constant shape
				# we can use as input to the model
				byte_statistics = example["byte_statistics"]

				# Group features and labels by session
				session_features, session_labels, session_metadata = [], [], []

				# Limit to only video flows
				all_flows = []
				for flow in byte_statistics[0].keys():
					video_ex = example["video_identification_features"][flow]
					try:
						tls_hostname = video_ex["tls_server_hostname"]
						service = service_tls_hostnames(tls_hostname)
						all_flows.append(flow)
					except KeyError:
						# some other website
						continue

				all_non_video_flows = get_difference(list(byte_statistics[0].keys()), all_flows)


				# Likely have a separate model for each service
				pred_service_type = 0 # SERVICE_TO_NUM[service] 

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
				# Flow acks maps flow -> u/d -> ack_n -> [times]
				# dup acks maps time instances to arr of length 2xself.n_flows
				# Indicating if any of the flows in the array experienced a duplicate ACK (loss)
				# the '2' is for u/d
				flow_acks = {flow: [{}, {}] for flow in all_flows}
				dup_acks_by_t = {i: [[], []] for i in range(n_bins)} # list of dup acks by time bin
				dup_acks = np.zeros((self.n_flows,n_bins,2))
				get_reqs_by_t = {i: [] for i in range(n_bins)}
				get_reqs = np.zeros((self.n_flows,n_bins,1))

				recent_window = 3

				for i in range(n_bins):
					sum_flows = []
					for flow in all_flows:
						if i < recent_window:
							s = 0
						else:
							s = i - recent_window + 1
						up_data = np.sum(byte_statistics[0][flow][s:i+1])
						down_data = np.sum(byte_statistics[1][flow][s:i+1])
						sum_flows.append(up_data + down_data)
						if flow[0] == flow[1]: # dummy filler flow:
							continue
						for j in range(2): # u/d
							for transp_data in byte_statistics[2+j][flow][i]:
								seq_n, ack_n, is_get_req = transp_data
								try:
									flow_acks[flow][j][ack_n].append(i)
									# duplicate ack
									try:
										dup_acks_by_t[i][j].append(flow)
									except KeyError:
										dup_acks_by_t[i][j] = [flow]
								except KeyError:
									flow_acks[flow][j][ack_n] = [i]
								# log that a get request occurred at this time for this flow (up-link)
								if is_get_req == 1 and j == 0:
									get_reqs_by_t[i].append(flow)


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
							# Populate transport features
							dup_acks[current_best_n[k],i,j] = dup_acks_by_t[i][j].count(all_flows[k])
						# Populate app-layer features
						get_reqs[current_best_n[k],i,0] += get_reqs_by_t[i].count(all_flows[k])
				# plt.imshow(get_reqs[:,:,0])
				# plt.show()

				size_byte_stats = byte_stats.shape
				to_viz, n_to_plot = [], 40
				total_bytes = np.cumsum(byte_stats,axis=1) # TODO - CHANGE THIS
				for i in range(bin_start, size_byte_stats[1] - self.history_length):
					if i % 10 == 0: continue
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
					get_request_indicator = self.get_recent_frames(i, get_reqs)
					other_temporal_features[:,:,4:5] = get_request_indicator


					# Clip between 0 and 1 (not guaranteed, but likely)
					temporal_image = temporal_image / self.max_dl

					# label for this time step
					labs = self.get_qoe_label(example,i - bin_start, _type)
					

					corr_check.append((labs, get_request_indicator))
					
					if labs is None:
						continue
					actual_labels = self.actual_labels

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
					session_labels.append(labs)
					session_metadata.append({
						"time_slot": i, # time slot in the capture
						"type": _type, # type of servie
						"actual_labels": actual_labels # un-discretized labels
					})
					
				
				self.X["all"].append(session_features)
				self.Y["all"].append(session_labels)
				self.metadata["all"].append(session_metadata)

		n_examples = len(self.X['all'])
		print("N sessions: {}".format(n_examples))
		n_train_examples = int(n_examples * .9)
		example_indices = {'train': None, 'val': None}
		example_indices['train'] = np.random.choice(range(n_examples), 
			size=n_train_examples, replace=False)
		example_indices['val'] = get_difference(list(range(n_examples)), example_indices['train'])
		for tv in example_indices:
			self.X[tv]['buffer_regression'] = [el for iex in example_indices[tv] for el in self.X['all'][iex] ]
			self.Y[tv]['buffer_regression'] = [el for iex in example_indices[tv] for el in self.Y['all'][iex] ]
			self.metadata[tv]['buffer_regression'] = [el for iex in example_indices[tv] for el in self.metadata['all'][iex]]

		n_row = 3
		n_back = [5,10,15,20,25,30]
		fig,ax = plt.subplots(n_row,n_row)
		for i in range(len(n_back)):
			plt_x,plt_y=[],[]
			for lab,gr in corr_check:
				if lab is None: continue
				plt_x.append(lab[0])
				plt_y.append(np.sum(np.sum(gr[:,-n_back[i]:,:])))
			print("N back: {}, min ngr: {}".format(n_back[i], np.min(plt_y)))
			print(np.sum(plt_y))
			row_i = i//n_row
			col_i = i%n_row
			ax[row_i,col_i].scatter(plt_x,plt_y)
			ax[row_i,col_i].set_title("{} back".format(n_back[i]))
		plt.show()
		exit(0)

	def predict_from_model(self, label_type, service_type, _x):
		"""Predict outputs based on inputs _x."""

		if self.models[label_type] is None:
			self.load_model(label_type)
		#print([np.sum(el[0:-1,:,:],axis=1) for el in _x])
		predicted_values = self.models[label_type].predict(_x)
		if label_type == "buffer_regression":
			predicted_values *= self.label_norm # un-normalize to the actual value
		else:
			raise ValueError("Label type {} not yet implemented in QOE_Classifier.predict_from_model.".format(label_type))

		return predicted_values

	def save_train_val(self):
		# saves the training and validation sets to pkls
		# each label_type type gets its own train + val set, since each will get its own classifier (at least for now)
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
			print(len(self.X['train']))
			self.X["val"], self.Y["val"], self.metadata['val'] = np.array(v['X']), np.array(v["Y"]), v['metadata']
			# Shuffle validation
			perm = list(range(len(self.X['val'])))
			perm = np.random.permutation(perm)
			self.X['val'] = self.X['val'][perm]
			self.Y['val'] = self.Y['val'][perm]
			self.metadata['val'] = [self.metadata['val'][i] for i in perm]
		else:
			raise ValueError("dtype {} not understood".format(dtype))

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

def main():
	#  Loads data & trains video QoE classifier of a certain type
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--sl', action='store',default="no")
	parser.add_argument('--pt', action='store', default='buffer_regression')
	parser.add_argument('--recover', action='store_true')
	args = parser.parse_args()

	br = Buffer_Regressor(skip_load=args.sl,recover_model=args.recover)
	problem_type = args.pt
	print("Trying problem type: {}".format(problem_type))
	br.run(problem_type)


if __name__ == "__main__":
	main()