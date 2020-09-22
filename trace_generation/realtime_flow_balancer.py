from subprocess import Popen, PIPE, STDOUT, call, check_output
from threading import Thread, Lock
import queue, time, os, signal, re, numpy as np, sys, traceback, select
from constants import *
from helpers import *
from abr_modeler import ABR_Modeler
from qoe_classifier import QOE_Classifier
from qoe_balancer import QOE_Balancer
from buffer_health_regressor import Buffer_Regressor

np.set_printoptions(threshold=np.inf,precision=3, suppress=True)


def remove_tshark_tmp_files():
	# tshark creates annoying tmp files that blow up -- delete them periodically
	try:
		call("sudo rm /tmp/wireshark*", shell=True)
	except FileNotFoundError:
		pass
def get_tshark_process(cmd):
	return Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)

 
INTERFACE = "nat0-eth0"

class thread_with_trace(Thread): 
	def __init__(self, *args, **keywords): 
		Thread.__init__(self, *args, **keywords) 
		self.killed = False
  
	def start(self): 
		self.__run_backup = self.run 
		self.run = self.__run       
		Thread.start(self) 
  
	def __run(self): 
		sys.settrace(self.globaltrace) 
		self.__run_backup() 
		self.run = self.__run_backup 
  
	def globaltrace(self, frame, event, arg): 
		if event == 'call': 
		  return self.localtrace 
		else: 
		  return None
	  
	def localtrace(self, frame, event, arg): 
		if self.killed: 
			if event == 'line': 
				raise SystemExit() 
		return self.localtrace 
	  
	def kill(self): 
		self.killed = True

class Proc_Out_Reader:
	def __init__(self, name):
		self.running = True
		self.name = name

	def enqueue_output(self, out, queue, poll_obj):
		while True:	
			# print("here in process: {}".format(self.name))
			if not self.running:
				break
			poll_result = poll_obj.poll(0)
			while poll_result:
				line = out.readline()
				queue.put(line.decode('utf-8').strip())
				poll_result = poll_obj.poll(0)
			time.sleep(.1)

class RealTime_Data_Collector:
	def __init__(self):

		# Wait until the interface of interest is up
		print("Waiting for Mininet interface to be created.")
		while not os.path.exists("/sys/class/net/{}".format(INTERFACE)):
			time.sleep(1)
		print("Interface created, proceeding.")

		self.tshark_packet_queue = {} # packets for each process
		self.tshark_p = {} # processes
		self.process_find_uids = { # grep commands used to find processes for refreshing
			"tls": "grep \"extensions_server_name\""
		}
		self.object_lock = Lock()
		# look for flow ID and TLS name; use grep to flter out non-TLS handshake packets
		self.tshark_tls_cmd = "sudo unbuffer tshark -Q -f \'tcp dst port 443\' -i {} -T fields -e ip.src -e ip.dst -e tcp.srcport"\
		" -e tcp.dstport -e ssl.handshake.extensions_server_name | unbuffer -p grep \"net\\|com\"".format(INTERFACE)
		self.tshark_flow_watch_cmd = "sudo unbuffer tshark -Q -i {}".format(INTERFACE)
		# either outgoing or incoming packet, filter on these two cases
		# Format gets src_ip, dst_ip, srcport, dst_ip,src_ip,srcport   (outgoing, incoming resp)
		self.tshark_flow_watch_cmd += " -f \'(src host {} and dst host {} and tcp src port {})  or (src host {} and dst host {} and tcp dst port {})\' -T fields"\
		" -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e tcp.ack -e ip.len"

		self.periods = {
			"remove_tmp_files": {
				"f": remove_tshark_tmp_files,
				"period": 20,
				"last": 0,
			},
			"restart_processes": {
				"f": self.refresh_tshark_instances,
				"period": 30,
				"last": 0,
			},
			"parse_statistics": {
				"f": self.parse_flow_statistics,
				"period": 1,
				"last": 0,
			},
			"create_features": {
				"f": self.create_features,
				"period": 3.5,
				"last": 0,
			},
			"predict_qoe": {
				"f": self.predict_qoe,
				"period": 1,
				"last": 0,
			},
			"balance_qoe": {
				"f": self.balance_qoe,
				"period": 5,
				"last": 0
			}
		}

		self.t_start = time.time()

		# object that holds players
		# IP -> {f_id: {byte_data, metadata} }
		self.players = {}
		self.threads = {} # output watchers

		# Features parameters
		self.feature_params = {
			"max_dl": BYTE_TRANSFER_NORM,
			"dup_ack_norm": DUP_ACK_NORM,
			"history_length": HISTORY_LENGTH,
			"n_flows": N_FLOWS,
			"allocation_amt": 10000,
			"total_n_channels": TOTAL_N_CHANNELS,
			"video_type_classes": {"twitch": 1, "netflix": 2, "youtube": 3}, # this needs to correlate with what we train it on
		}

		# Prediction parameters
		self.abr_modeler = ABR_Modeler()
		self.qoe_classifier = QOE_Classifier()
		self.buffer_regressor = Buffer_Regressor()

		def get_predict_fn(qoe_metric, service_type):
			# if qoe_metric == "buffer":
			# 	return lambda features : self.qoe_classifier.predict_from_model("buffer", service_type, features)
			if qoe_metric == "buffer":
				return lambda features : self.buffer_regressor.predict_from_model("buffer_regression", service_type, features)
			elif qoe_metric == "bitrate":
				return lambda features : self.abr_modeler.predict_abr(service_type, features)
			else:
				raise ValueError("QoE metric {} not yet implemented.".format(qoe_metric))

		self.prediction_metrics = ["buffer", "bitrate"]
		self.prediction_models = {
			qoe_m: {
				s: get_predict_fn(qoe_m, s) 
				for s in VIDEO_SERVICES
			}
			for qoe_m in self.prediction_metrics
		}

		# QoE balancing parameters
		self.qoe_balancer = QOE_Balancer()
		self.default_available_bw = AVAILABLE_BW

	def add_service_flow(self, src_ip, service, f_id):
		try:
			self.players[src_ip]
		except:
			self.players[src_ip] = {
				"features": {pm: None for pm in self.prediction_metrics}, 
				"flows": {}, 
				"service": service, # NOTE -- THIS WILL NEED TO CHANGE FOR A ROBUST SOLUTION
				"metadata": {
					"ordered_flows": [],
					"current_best_flows": {},
					"t_last_update": time.time(),
				},
				"predictions": {pm: [] for pm in self.prediction_metrics},
				"bandwidth_allocations": [],
				"t_added": time.time(),
		}
		self.players[src_ip]["metadata"]["ordered_flows"].append(f_id)
		self.players[src_ip]["flows"][f_id] = {
			"byte_data": [[], []], 
			"metadata": {
				"service": service,
				"acks": {},
			}, 
			"statistics": None, 
		}
		ip, port = f_id
		self.spawn_tshark("flow_watch", src_ip = src_ip, dst_ip = ip, src_port = port)

	def balance_qoe(self):
		# sets up the player objects for the qoe balancer
		# calls the optimization to obtain bandwidth allocations
		# sets bandwidths obtained by the optimizer
		# adds these bandwidths to memory, to be used by other components of the process
		player_keys = [p for p in self.players if\
			self.players[p]["predictions"]["bitrate"] != []\
			and self.players[p]["predictions"]["buffer"] != []]
		if player_keys == []: return
		players = [{
			"bitrate": self.players[p]["predictions"]["bitrate"][-1][1],
			"buffer": self.players[p]["predictions"]["buffer"][-1][1],
			"service": self.players[p]["service"],
		} for p in player_keys]
		self.qoe_balancer.set_players(players)
		optimal_bandwidth_allocation = self.qoe_balancer.solve_for_slices()
		self.set_bandwidth(player_keys, optimal_bandwidth_allocation)

		for p, bw_alloc in zip(player_keys, optimal_bandwidth_allocation):
			self.players[p]["bandwidth_allocations"].append((time.time(), bw_alloc))

	def create_features(self):
		# for each user, updates player_features 
		# to contain features used for QoE classification

		# Features can be of multiple types, for a variety of algorithms

		current_t_bin = int((time.time() - self.t_start) / T_INTERVAL)
		large_n = self.feature_params["allocation_amt"]
		allocation_amt = np.maximum(2*current_t_bin, large_n)
		history_length = self.feature_params["history_length"]
		n_flows = self.feature_params["n_flows"]
		total_n_channels = self.feature_params["total_n_channels"]
		for player in self.players: # I don't think I need to lock this, since this is in the main thread
			# statistics
			player_flows = self.players[player]["metadata"]["ordered_flows"]
			t_last_update = self.players[player]["metadata"]["t_last_update"]
			starting_bin = int(np.floor((t_last_update - 
					self.t_start) / T_INTERVAL))
			if len(player_flows) == 0: continue
			# this is the same for all flows now, BUT NEEDS TO CHANGE
			# when we move to NATs or players changing services, etc..
			service = self.players[player]["service"]
			for f_id in player_flows:
				this_f = self.players[player]["flows"][f_id]
				# update indices corresponding to how much time
				# has elapsed since we last constructed features
				# add 1, since the border region may not have been 
				# complete at the time
				
				if this_f["statistics"] is None:
					# pre-allocate some space for this flow
					this_f["statistics"] = np.zeros((allocation_amt, 4)) # u/d bytes/acks
				elif (.75 * this_f["statistics"].shape[0]) < current_t_bin:
					# allocate more space
					this_f["statistics"] = np.concatenate([this_f["statistics"], 
						np.zeros((allocation_amt, 4))], axis=1)

				# update statistics object
				for i in range(2): # u/d
					for report in this_f["byte_data"][i]:
						pkt_time, pkt_len, ack_n = report
						this_bin = int((pkt_time - self.t_start) / T_INTERVAL)
						this_f["statistics"][this_bin,i] += pkt_len
						try:
							# TODO -- make sure wrap-around isn't a problem
							this_f["metadata"]["acks"][ack_n] 
							this_f["statistics"][this_bin, i+2] += 1 # duplicate ACK
						except KeyError:
							this_f["metadata"]["acks"][ack_n] = None
					this_f["byte_data"][i] = [] # clear this, to prevent build-up

				
			
			if self.players[player]["features"]["buffer"] is None:
				# Initialize features if needed
					self.players[player]["features"]["buffer"] = np.zeros((n_flows,history_length,total_n_channels))
					previous_buffer_features = self.players[player]["features"]["buffer"][:,:,:]
			else:
				# grab features from previous feature creation round (removing expert features)
				previous_buffer_features = self.players[player]["features"]["buffer"][:-1,:,:]
			# Left shift features to make room for new information
			current_bin = int(np.ceil((time.time() - # bin we end at (fill in up to this)
					self.t_start) / T_INTERVAL))
			# When calculating n_to_shift, we count down from history_length
			# since we have the newest information at the last index
			n_indices_to_update = (current_bin - starting_bin)
			n_to_shift = history_length - n_indices_to_update + 1
			previous_buffer_features = np.roll(previous_buffer_features, 
				shift=n_to_shift, axis=1)
			# write over the old information with zeros
			previous_buffer_features[:,-n_indices_to_update:,:] = np.zeros((n_flows,n_indices_to_update,total_n_channels))

			# Un-normalize, since we update raw values
			previous_buffer_features[:,:,0:2] *= self.feature_params["max_dl"]
			previous_buffer_features[:,:,2:4] *= self.feature_params["dup_ack_norm"]
			previous_buffer_features[:,:,4:6] *= (10 * self.feature_params["max_dl"])
			for ti in range(n_indices_to_update):
				sum_flows = []
				for flow in player_flows:
					# calculate which flows are the best
					if starting_bin + ti < history_length:
						up_data = np.sum(self.players[player]["flows"][flow]["statistics"][:starting_bin+ti+1,0])
						down_data = np.sum(self.players[player]["flows"][flow]["statistics"][:starting_bin+ti+1,1])
					else:
						up_data = np.sum(self.players[player]["flows"][flow]["statistics"][starting_bin+ti-history_length+1:current_t_bin+1,0])
						down_data = np.sum(self.players[player]["flows"][flow]["statistics"][starting_bin+ti-history_length+1:current_t_bin+1,1])
					sum_flows.append(up_data+down_data)
				if len(sum_flows) < n_flows:
					# append dummy info
					[sum_flows.append(-1) for _ in range(n_flows - len(sum_flows))]
				best_n = np.argpartition(sum_flows,-1*n_flows)[-1*n_flows:]
				if self.players[player]["metadata"]["current_best_flows"] == {}:
					# initialize
					self.players[player]["metadata"]["current_best_flows"] = {best: i for i,best in enumerate(best_n)}
				previous_best_n = self.players[player]["metadata"]["current_best_flows"]
				if set(best_n) != set(previous_best_n.keys()):
					# Update the array with the new best flows
					new_best_flows = get_difference(best_n, previous_best_n.keys())
					flows_to_remove = get_difference(previous_best_n.keys(), best_n)
					for add_flow, remove_flow in zip(new_best_flows, flows_to_remove):
						i_of_new_flow = previous_best_n[remove_flow]
						del previous_best_n[remove_flow]
						previous_best_n[add_flow] = i_of_new_flow	
				current_best_n = previous_best_n
				for flow_i in best_n:
					try:
						flow = player_flows[flow_i]
					except IndexError: # dummy information
						continue
					i_in_features = current_best_n[flow_i]
					# print("Forming features for active flow: {}".format(flow))
					this_flow_data = self.players[player]["flows"][flow]["statistics"]
					# print(this_flow_data[0:30,0:2])
					# Populate current time bin data (byte stats and dup acks)
					this_t_data = this_flow_data[starting_bin+ti,:]
					previous_buffer_features[i_in_features,-n_indices_to_update+ti,0:4] = this_t_data
					# Update cumulative byte transfers (previous cumulative plus current stats)
					previous_buffer_features[i_in_features,-n_indices_to_update+ti,4] = previous_buffer_features[i_in_features,-n_indices_to_update+ti-1,4] +\
						previous_buffer_features[i_in_features,-n_indices_to_update+ti,0]
					previous_buffer_features[i_in_features,-n_indices_to_update+ti,5] = previous_buffer_features[i_in_features,-n_indices_to_update+ti-1,5] +\
						previous_buffer_features[i_in_features,-n_indices_to_update+ti,1]
			previous_buffer_features[:,:,6] = (previous_buffer_features[:,:,0] > GET_REQUEST_SIZE).astype(np.int64)

			# Roughly normalize byte transfers between 0 and 1
			previous_buffer_features[:,:,0:2] /= self.feature_params["max_dl"]
			previous_buffer_features[:,:,4:6] /= (10 * self.feature_params["max_dl"])
			# Normalize dup ack features
			previous_buffer_features[:,:,2:4] /= self.feature_params["dup_ack_norm"]
			self.players[player]['metadata']['t_last_update'] = time.time()
			# print(player)
			# for i in range(TOTAL_N_CHANNELS):
			# 	print(np.sum(previous_buffer_features[:,:,i],axis=1))

			# Create expert features
			pred_service_type = self.feature_params["video_type_classes"][service]
			expert_buffer_features = np.zeros((1,history_length,total_n_channels))
			expert_buffer_features[0,0,0] = int((time.time() - self.players[player]["t_added"]) / T_INTERVAL) / MAX_TIME
			expert_buffer_features[0,1,0] = 0 # pred_service_type

			buffer_features = np.concatenate([previous_buffer_features, expert_buffer_features], axis=0)
			self.players[player]["features"]["buffer"] = buffer_features

			# Create bitrate estimation features
			# this is [last_n_bandwidths, buffer_health @ time=t where t satisfies: buffer_health(t) + time(t) = time.time()] 
			if self.players[player]["predictions"]["buffer"] != []:
				diffs = [np.abs(time.time() - report_t - bh) for report_t, bh in self.players[player]["predictions"]["buffer"]]
				report_of_interest_i = np.argmin(diffs) # could be slow for long sessions, might want to do binary search
				time_of_report, buffer_health_at_report = self.players[player]["predictions"]["buffer"][report_of_interest_i]
			else:
				time_of_report = time.time()
				# set default buffer of 0s; a conservative estimate given we have no information about the user
				buffer_health_at_report = 0
			last_n_bw = self.get_last_n_bw(player, time_of_report)
			bitrate_features = last_n_bw + [buffer_health_at_report]
			self.players[player]["features"]["bitrate"]	= bitrate_features

	def get_last_n_bw(self, player, t):
		# Number of previous bandwidths to obtain
		# Bandwidths are assumed to be spaced at the interval during training
		# right now, this is set in throughput_traces/traces to be intervals of 5s
		# t is sometime in the recent past
		n_to_get = self.abr_modeler.abr_last_n_bw

		# So, we obtain the last n_to_get bandwidths starting before t set by us, 
		# spaced by 5 seconds
		# 5 is a constant set by our bandwidth traces (could change in future)
		# i.e. we train on intervals of 5 seconds

		# Returned bw is in kbps

		ret_bw = []
		ts = [t - 5*i for i in range(n_to_get)]
		for _t in ts:
			# Look for bw right before this time
			for t_bw, bw in reversed(self.players[player]["bandwidth_allocations"]):
				if _t - t_bw > 0:
					ret_bw.append(bw / 1000)
					break
		while len(ret_bw) < n_to_get:
			# append with default values for available bw 
			# this should only happen for new flows without information
			ret_bw.append(self.default_available_bw / 1000)

		return ret_bw

	def parse_flow_statistics(self):
		# Prints information about flows
		for player in self.players:
			#print("Player: {}".format(player))
			for flow in self.players[player]["flows"]:
				this_flow = self.players[player]["flows"][flow]
				# print("Flow : {} data up: {}, data down: {}".format(
				# 	flow, sum(el[1] for el in this_flow["byte_data"][0]), 
				# 	sum(el[1] for el in this_flow["byte_data"][1])))
			#print("\n")

	def read_tshark(self, process_names=None):
		# Read all output from all processes
		if process_names is None:
			process_names = self.tshark_p.keys()
		for process_key in process_names:
			try:
				self.tshark_packet_queue[process_key]
			except KeyError:
				try:
					self.object_lock.acquire()
					self.tshark_packet_queue[process_key] = queue.Queue()
				except:	
					print('Hit an error acquiring lock for new packet queue key in read_tshark -- {}'.format(sys.exc_info()))
					exit(0)
				finally:
					self.object_lock.release()
			por = Proc_Out_Reader(process_key)
			poll_obj = select.poll()
			poll_obj.register(self.tshark_p[process_key]["process"].stdout, select.POLLIN)
			self.threads[process_key]["reader"] = por
			por.enqueue_output(self.tshark_p[process_key]["process"].stdout, 
				self.tshark_packet_queue[process_key],
				poll_obj)
			
	def refresh_tshark_instances(self):
		# tshark tends to eat up memory by storing temporary variables
		# to counter-act this while retaining all of tsharks nice features, we restart the process periodically
		

		# THIS IS A LITTLE SLOW
		for process_key in self.tshark_p:
			this_process = self.tshark_p[process_key]
			if time.time() - this_process["t_start"] > this_process["ttl"]:
				print("Refreshing process: {}".format(process_key))
				old_processes = check_output("ps ax -o pid= -o cmd= | {}".format(self.process_find_uids[process_key]), shell=True).decode('utf-8').split("\n")
				# start a new, identical process
				new_process = get_tshark_process(this_process["cmd"])
				
				# replace the old process with the new one
				self.tshark_p[process_key] = {
					"process": new_process,
					"ttl": this_process["ttl"],
					"t_start": time.time(),
					"cmd": this_process["cmd"],
				}
				# # restart the thread watcher
				# self.threads[process_key]["reader"].running = False
				# # wait for the thread to stop
				# print('waiting for thread to stop')
				# self.threads[process_key]["thread"].join()
				# print('done waiting')
				# print("killing process")
				# self.threads[process_key]["thread"].kill()
				self.threads[process_key]["reader"].running = False
				print("joining process")
				self.threads[process_key]["thread"].join()

				# kill this tshark process
				cmd = ""
				for tshark_process in old_processes:
					if "pid" in tshark_process: continue
					tshark_process = tshark_process.strip()
					if tshark_process == "": continue
					try:
						proc_info = re.search("(\d+) \D(.+)", tshark_process)
						pid = proc_info.group(1)
						cmd += "sudo kill -9 {} &;".format(pid)
					except AttributeError:
						print("Error -- attribute error: " + tshark_process)
				
				# start a new thread to watch this tshark instance
				self.threads[process_key]["thread"] = Thread(target=self.read_tshark, kwargs={"process_names":[process_key]}, daemon=True)
				self.threads[process_key]["thread"].start()

	
	def set_bandwidth(self, player_keys, optimal_bandwidth_allocation):
		i = 0
		call("tcdel {} --all".format(INTERFACE), shell=True)
		for pk, bwa in zip(player_keys, optimal_bandwidth_allocation):
			bwa = np.maximum(int(bwa / 1000), 100) # allow a trickle of data 
			print("Seting player {} BW to {} Kbps".format(pk, bwa))
			if i == 0:
				call("tcset {} --network {}/32 --rate {}Kbps &".format(
					INTERFACE, pk, bwa), shell=True)
			else:
				call("tcset {} --network {}/32 --rate {}Kbps --add &".format(
				INTERFACE, pk, bwa), shell=True)
			i +=1

	def spawn_tshark(self, _type, **kwargs):
		# one master tshark listener that watches for TLS hostnames
		# worker tshark listeners that listen to individual flows
		if _type == "tls":
			self.tshark_p["tls"] =  {
				"process": get_tshark_process(self.tshark_tls_cmd),
				"ttl": 100,
				"t_start": time.time(),
				"cmd": self.tshark_tls_cmd,
			}
		elif _type == "flow_watch":
			src_port = kwargs["src_port"]
			src_ip = kwargs["src_ip"]
			dst_ip = kwargs["dst_ip"]
			cmd = str(self.tshark_flow_watch_cmd.format(src_ip, dst_ip, src_port, dst_ip,src_ip, src_port))
			self.tshark_p[dst_ip, src_port] = {
				"process": get_tshark_process(cmd),
				"ttl": 200,
				"t_start": time.time(),
				"cmd": cmd,
			}
			self.threads[dst_ip, src_port] = {
				"thread" : Thread(target=self.read_tshark, 
				kwargs={"process_names": [(dst_ip,src_port)]}, daemon=True),
				"reader": None,
			}
			self.threads[dst_ip, src_port]["thread"].start()
			self.process_find_uids[dst_ip, src_port] = "grep -E \"{}.*{}\"".format(dst_ip, src_port)

	def predict_qoe(self):
		"""Predicts qoe metrics of interest based on current features, 
			and updates them in the players object."""
		for prediction_metric in self.prediction_metrics:
			for service in VIDEO_SERVICES:
				these_players = [player for player in self.players if\
					self.players[player]['service'] == service and self.players[player]['features'][prediction_metric] is not None]
				# Predictions are run in parallel, since this is fastest
				all_player_features = [self.players[player]["features"][prediction_metric] \
					for player in these_players]
				
				if all_player_features == []: continue

				# Call the prediction function
				if prediction_metric == 'bitrate':
					print(all_player_features)
				predicted_metrics = self.prediction_models[prediction_metric][service](np.array(all_player_features))
				print(predicted_metrics)
				# save predictions for other parts of the pipeline
				for predicted_metric, player in zip(predicted_metrics, these_players):
					self.players[player]["predictions"][prediction_metric].append((time.time(), predicted_metric))

	def process_packets(self):
		try:
			self.object_lock.acquire()
			process_keys = list(self.tshark_packet_queue.keys())
		except:
			print('Hit error in process_packets lock acquire -- {}'.format(sys.exc_info()))
			exit(0)
		finally:
			self.object_lock.release()
		for process_key in process_keys:
			while not self.tshark_packet_queue[process_key].empty():
				packet = self.tshark_packet_queue[process_key].get(timeout=.1)
				# do stuff with this
				# tls -- spawn tshark session watching this flow and create data structures
				# regular flow update -- add statistics
				if packet[0:7] == "Running": continue # first line
				try:
					if process_key == "tls":
						# check to see if this is a new video flow
						src, dst, srcport, dstport, tls_hostname = packet.split("\t")
						try:
							service = service_tls_hostnames(tls_hostname)
							self.add_service_flow(src, service, (dst,srcport))
						except KeyError:
							# some other website
							continue
					else:
						src_ip, dst_ip, src_port, dst_port, ack_num, pkt_len = packet.split("\t")
						if is_internal(src_ip):
							ud = 0
							player_ip = src_ip
							f_id = (dst_ip, src_port)
						else:
							ud = 1
							player_ip = dst_ip
							f_id = (src_ip, dst_port)
						self.players[player_ip]["flows"][f_id]["byte_data"][ud].append(
							[time.time(), int(pkt_len), int(ack_num)])
				except:
					print("Error parsing packet: {}".format(packet))
					print(traceback.format_exc())

	def loop_packet_processor(self):
		i=0
		while True:
			self.process_packets() # process these packets
			for periodic_check in self.periods:
				if time.time() - self.periods[periodic_check]["last"] > self.periods[periodic_check]["period"]:
					self.periods[periodic_check]["f"]()
					self.periods[periodic_check]["last"] = time.time()
			i+=1
			time.sleep(.05)

def main():
	rdc = RealTime_Data_Collector()
	t1 = Thread(target=rdc.read_tshark, kwargs={"process_names":["tls"]}, daemon=True)
	t2 = Thread(target=rdc.loop_packet_processor)
	rdc.threads["tls"] = {"thread": t1, "reader": None}
	rdc.threads["main"] = {"thread": t2, "reader": None}
	rdc.spawn_tshark("tls")
	t1.start()
	t2.start()

if __name__ == "__main__":
	main()