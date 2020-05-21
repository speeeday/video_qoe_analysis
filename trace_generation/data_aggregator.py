import numpy as np, os, glob, pickle, dpkt, re, csv, socket, json
from subprocess import call, check_output
import geoip2.database
import matplotlib
import matplotlib.pyplot as plt

from constants import *
from helpers import *

def to_ip_str(ip_bytes):
	return socket.inet_ntoa(ip_bytes)

# general goal is to create train/val datasets for features -> V/NV and features -> QoE of various types
# QoE is state, quality of stream playing, buffer warning/level

class Data_Aggregator:
	def __init__(self, _type):
		self.type = _type
		self.log_dir = "./logs"
		self.pcap_dir = "./pcaps"
		self.save_dir = "./features"
		self.metadata_dir = METADATA_DIR
		self.error_report_dir = ERROR_REPORT_DIR
		self.t_interval = T_INTERVAL # seconds, interval over which to bin/sum/average statistics like # bytes received
		self.n_ips = 5
		self.history_length = 30

		self.known_ip_list_fn = KNOWN_IP_LIST_FN

		self.qoe_features = {}

		self.append = True # whether or not we should append the aggregated features to the current features files or not
		self.stats_panel_info = None
		self.t_start_recording_offset = 0

	def cleanup_files(self):
		# removes log files, pcaps, as they are no longer useful

		call("rm pcaps/{}*".format(self.type), shell=True)
		call("rm logs/{}*".format(self.type), shell=True)

	def is_internal(self, ip):
		# for now, check to see if ip is in constant field INTERNAL_IPS
		# later, possibly check against a radix tree for the network of interest
		return ip in INTERNAL_IPS

	def load_data(self, link, _id):
		"""Loads capture statistics and pcaps for a video."""
		# load advanced panel statistics
		if self.type != "no_video":
			# per-flow video classifier feature creator
			self.per_flow_video_classifier(_id)

			stats_file = os.path.join(self.log_dir, "{}_stats_log_{}-{}-stats.csv".format(self.type,_id,link))
			stats = []
			with open(stats_file, 'r') as f:
				csvr = csv.DictReader(f)
				[stats.append(row) for row in csvr]
			self.stats_panel_info = stats
			self.t_start_recording = stats[0]["timestamp"]		

			# load video stats data
			meta_data_file = os.path.join(self.log_dir, "{}_stats_log_{}-{}-metadata.pkl".format(self.type,_id,link))
			meta_data = pickle.load(open(meta_data_file,'rb'))
			if self.type != "no_video":
				self.t_start_recording_offset = meta_data['start_wait']
			else:
				self.t_start_recording_offset = 0
		else:
			self.video_identification_features = None
		# load relevant pcap data
		self.load_pcap(_id)

	def load_pcap(self, _id):
		pcap_file_name = os.path.join(self.pcap_dir, "{}_{}.pcap".format(self.type,_id))
		# first, convert the file to the correct format
		call("mv {} {}".format(pcap_file_name, pcap_file_name + "ng"), shell=True)
		call("editcap -F libpcap -T ether {} {}".format(pcap_file_name + "ng", pcap_file_name), shell=True)
		call("rm {}".format(pcap_file_name + "ng"), shell=True)
		pcap_file = dpkt.pcap.Reader(open(pcap_file_name,'rb'))
		first = last = next(pcap_file, None)
		for last in pcap_file:
		    pass
		t_start,_ = first
		t_end,_ = last
		total_time = t_end - t_start
		n_bins = int(np.ceil(total_time / self.t_interval))
		self.n_bins = n_bins
		self.bytes_transfered = [{}, {}] # up, down, each of which are dicts: dst/src ip -> [bytes received during time chunks]

		pcap_file = dpkt.pcap.Reader(open(pcap_file_name,'rb'))
		port_amts = {}
		for ts, buf in pcap_file:
			eth_pkt = dpkt.ethernet.Ethernet(buf)
			if eth_pkt.type != dpkt.ethernet.ETH_TYPE_IP:
				continue

			ip_pkt = eth_pkt.data
			if ip_pkt.p != dpkt.ip.IP_PROTO_TCP and ip_pkt.p != dpkt.ip.IP_PROTO_UDP:
				# make sure its UDP or TCP
				continue

			app_data = ip_pkt.data

			# TODO - make this as if we're looking at a network of networks
			# each (src_ip, source port) is a user stream of data
			# once a user stream has been identified as containing video, we estimate QoE (potential refinement of feature set)

			src_ip = to_ip_str(ip_pkt.src)
			dst_ip = to_ip_str(ip_pkt.dst)

			# for now just exclude things that aren't https
			source_port = app_data.sport 
			dest_port = app_data.dport
			if source_port != 443 and dest_port != 443:
				continue
			

			if self.is_internal(src_ip):
				# outgoing packet
				try:
					self.bytes_transfered[0][dst_ip]
				except KeyError:
					self.bytes_transfered[0][dst_ip] = {}#np.zeros(n_bins)
				try:
					self.bytes_transfered[0][dst_ip][source_port]
				except KeyError:
					self.bytes_transfered[0][dst_ip][source_port] = np.zeros(n_bins)
				bin_of_interest = int(np.floor((ts - t_start / self.t_interval)))
				self.bytes_transfered[0][dst_ip][source_port][bin_of_interest] += ip_pkt.len

			elif self.is_internal(dst_ip):
				# incoming packet
				try:
					self.bytes_transfered[1][src_ip]
				except KeyError:
					self.bytes_transfered[1][src_ip] ={}# np.zeros(n_bins)
				try:
					self.bytes_transfered[1][src_ip][dest_port]
				except KeyError:
					self.bytes_transfered[1][src_ip][dest_port] = np.zeros(n_bins)
				bin_of_interest = int(np.floor((ts - t_start / self.t_interval)))
				self.bytes_transfered[1][src_ip][dest_port][bin_of_interest] += ip_pkt.len
			else:
				print("Neither src: {} nor dst: {} are internal...".format(src_ip, dst_ip))
		# place all IPs into up and down, for convenience
		all_flows = []
		for i in range(2):
			for ip in self.bytes_transfered[i]:
				for port in self.bytes_transfered[i][ip]:
					try:
						self.bytes_transfered[1-i][ip]
					except KeyError:
						self.bytes_transfered[1-i][ip] = {}
					try:
						self.bytes_transfered[1-i][ip][port]
					except KeyError:
						self.bytes_transfered[1-i][ip][port] = np.zeros(n_bins)

		# for flow in all_ips:
		# 	for i in [0,1]:
		# 		try:
		# 			self.bytes_transfered[i][ip]
		# 		except KeyError:
		# 			self.bytes_transfered[i][ip] = np.zeros(self.n_bins)
		self.visualize_bit_transfers(_id)
		pickle.dump(self.bytes_transfered, open(os.path.join(self.pcap_dir, "{}_processed.pkl".format(_id)),'wb'))

	def per_flow_video_classifier(self, _id):
		pcap_file_name = os.path.join(self.pcap_dir, "{}_{}.pcap".format(self.type,_id))
		decrypted_pkt_data = check_output("tshark -o ssl.keylog_file:{} -r {} -Y tcp.port==443 -V -T json".format(
			SSL_KEYLOG_FILE, pcap_file_name), shell=True)
		def dict_raise_on_duplicates(ordered_pairs):
			# handles duplicate keys, which of course happens for the field we care about
			d = {}
			for k,v in ordered_pairs:
				if k == "http2.header":
					if k in d:
						d[k].append(v)
					else:
						d[k] = [v]
				else:
					d[k] = v
			return d
		tester_function = {
			"twitch": lambda uri: "v1/segment" in uri,
			"youtube": lambda uri: "videoplayback?" in uri,
			"netflix": lambda uri: "range/" in uri,
		}[self.type]
		def is_http2_request_for_video(pkt):
			try:
				http_info = pkt["http2"]
			except KeyError:
				# No http info, continueq
				return None
			try:
				http_info = http_info["http2.stream"]
			except TypeError:
				return None
			try:
				http_headers = http_info["http2.header"]
			except KeyError:
				return None
			for http_header in http_headers:
				try:
					requested_uri = http_header["http2.header.value"]
				except KeyError:
					continue
				if tester_function(requested_uri):
					return True
			return None
		def is_http_request_for_video(pkt):
			try:
				http_info = pkt["http"]
			except KeyError:
				return None
			try:
				requested_uri = http_info["http.request.full_uri"]
			except KeyError:
				return None
			if tester_function(requested_uri):
				return True
			return None
		def get_tls_server_hostname(pkt):
			try:
				handshake_info = pkt["ssl"]["ssl.record"]["ssl.handshake"]
			except (KeyError, TypeError):
				return None
			for k in handshake_info:
				if "Extension: server_name" in k:
					try:
						server_name = handshake_info[k]["Server Name Indication extension"]["ssl.handshake.extensions_server_name"]
					except KeyError:
						continue
					return server_name
			return None

		def get_flow(pkt):
			dst_ip = pkt["ip"]["ip.dst"]
			src_port = int(pkt["tcp"]["tcp.srcport"])
			dst_port = int(pkt["tcp"]["tcp.dstport"])
			flow_id = (dst_ip, src_port, dst_port)
			return flow_id

		decrypted_pkt_data = json.loads(decrypted_pkt_data.decode('utf-8'), object_pairs_hook=dict_raise_on_duplicates)
		flow_descriptors = {}
		all_flows = []
		for pkt in decrypted_pkt_data:
			pkt = pkt["_source"]["layers"]
			# Get label
			# Flow ID
			flow_id = get_flow(pkt)
			all_flows.append(flow_id)
			if is_http2_request_for_video(pkt) or is_http_request_for_video(pkt):
				try:
					flow_descriptors[flow_id]
				except KeyError:
					flow_descriptors[flow_id] = {
						"is_video": True,
						"total_bytes": 0,
						"total_bytes_up": 0,
						"total_bytes_down": 0,
						"byte_deliveries": [] # (time, n_bytes) for packet deliveries
					}
		all_flows = set(all_flows)
		not_video = get_difference(all_flows, list(flow_descriptors.keys()))
		for k in not_video:
			flow_descriptors[k] = {
				"is_video": False,
				"total_bytes": 0,
				"total_bytes_up": 0,
				"total_bytes_down": 0,
				"byte_deliveries": [] # (time, n_bytes) for packet deliveries
			}
		
		# Get all the features -- note, we need to be able to get all these features 
		# from encrypted packets
		for pkt in decrypted_pkt_data:
			pkt = pkt["_source"]["layers"]
			flow_id = get_flow(pkt)
			pkt_size = int(pkt["frame"]["frame.len"])
			flow_descriptors[flow_id]["total_bytes"] += pkt_size
			if flow_id[0] in INTERNAL_IPS:
				# delivery
				pkt_time = float(pkt["frame"]["frame.time_epoch"])
				flow_descriptors[flow_id]["byte_deliveries"].append((pkt_time, pkt_size))
				flow_descriptors[flow_id]["total_bytes_down"] += pkt_size
			else:
				flow_descriptors[flow_id]["total_bytes_up"] += pkt_size

			# see if this is a tls handshake packet
			# if so, get the tls server hostname
			tls_server_hostname = get_tls_server_hostname(pkt)
			if tls_server_hostname:
				flow_descriptors[flow_id]["tls_server_hostname"] = tls_server_hostname
		
		# calculate throughput-based features
		duration = 5 # seconds
		def get_throughputs(packet_deliveries):
			times = np.array([el[0] for el in packet_deliveries])
			sizes = np.array([el[1] for el in packet_deliveries])
			times = times - np.min(times)
			n_bins = int(np.ceil(np.max(times) / duration))
			throughputs = np.zeros(n_bins)
			for t,s in zip(times, sizes):
				_bin = int(t//duration)
				throughputs[_bin] += s
			throughputs /= duration
			return throughputs

		for flow in flow_descriptors:
			byte_deliveries = flow_descriptors[flow]["byte_deliveries"]
			if len(byte_deliveries) <= 1:
				continue
			tpt_measurements = get_throughputs(byte_deliveries)
			flow_descriptors[flow]["mean_throughput"] = np.mean(tpt_measurements)
			flow_descriptors[flow]["max_throughput"] = np.max(tpt_measurements)
			flow_descriptors[flow]["var_throughput"] = np.var(tpt_measurements)
		# Save these features with the rest
		self.video_identification_features = flow_descriptors

	def populate_features(self, link, _id):
		self.qoe_features[_id] = {
			"byte_statistics": None,
			"other_statistics": {},
			"info": {"link": link},
			"stats_panel": self.stats_panel_info,
			"start_offset": self.t_start_recording_offset,
			"video_identification_features": self.video_identification_features,
		}

		# other statistics
		# distribution of ASN's communicated with up to current timestep
		# # of ASNs communicated with up to current timestep
		# is dst IP (/24) known to be associated with video services (which)?
		all_ips = list(self.bytes_transfered[0].keys())
		asns = {}
		with geoip2.database.Reader(os.path.join(self.metadata_dir,"GeoLite2-ASN.mmdb")) as reader:
			for ip in all_ips:
				try:
					response = reader.asn(ip)
				except geoip2.errors.AddressNotFoundError:
					continue
				#asn = response.autonomous_system_number
				asn = response.autonomous_system_organization

				try:
					asns[asn] += 1 # could weight by the amount of traffic
				except KeyError:
					asns[asn] = 1
		n_asns = len(asns)
		asn_dist = get_asn_dist(asns)
		self.qoe_features[_id]["other_statistics"]["asn_dist"] = asn_dist
		self.qoe_features[_id]["other_statistics"]["n_total_asns"] = n_asns
		self.qoe_features[_id]["other_statistics"]["ip_likelihood"] = get_ip_likelihood(all_ips, self.type, modify=False)

		# byte statistics

		# create an array n_ips x history_length x 2 where the statistics are shown for the 
		# ips with the most traffic (traffic is sum of up and down) over the last history_length interval

		# we don't start recording videco stats until a certain time, fastforward here
		if self.type == "no_video":
			bin_start = 0
		else:
			bin_start = int(np.floor(self.t_start_recording_offset / self.t_interval))
		byte_stats = np.zeros((self.n_ips, self.n_bins, 2)) # 
		current_best_n = {} # dict with self.n_ips keys; each key is index of ip in all_ips -> row in byte_stats this flow occupies
		all_ports = set([port for ip in self.bytes_transfered[0] for port in self.bytes_transfered[0][ip]])

		# NOTE - I effectively zero-out all packet transfers up to the point where I start recording statistics
		for i in range(bin_start, self.n_bins):
			sum_ips = np.array([sum([sum(self.bytes_transfered[0][ip][port][i-self.history_length:i]) for port in self.bytes_transfered[0][ip]])
			 + sum([sum(self.bytes_transfered[1][ip][port][i-self.history_length:i]) for port in self.bytes_transfered[1][ip]]) for ip in all_ips])
			# get max n
			try:
				best_n = np.argpartition(sum_ips,-1*self.n_ips)[-1*self.n_ips:]
			except ValueError:
				print("Link: {}, ID: {}, IPS: {}".format(link,_id,all_ips)); exit(0)
			if current_best_n == {}:
				current_best_n = {best: i for i,best in enumerate(best_n)}
			if set(best_n) != set(current_best_n.keys()):
				new_best_flows = get_difference(best_n, current_best_n.keys())
				flows_to_remove = get_difference(current_best_n.keys(), best_n)
				for add_flow, remove_flow in zip(new_best_flows, flows_to_remove):
					i_of_new_flow = current_best_n[remove_flow]
					del current_best_n[remove_flow]
					current_best_n[add_flow] = i_of_new_flow

			for ip_to_include in best_n:
				byte_stats[current_best_n[ip_to_include]][i][0] = sum([self.bytes_transfered[0][all_ips[ip_to_include]][port][i] for port in self.bytes_transfered[0][all_ips[ip_to_include]]])
				byte_stats[current_best_n[ip_to_include]][i][1] = sum([self.bytes_transfered[1][all_ips[ip_to_include]][port][i] for port in self.bytes_transfered[1][all_ips[ip_to_include]]])

		self.qoe_features[_id]["byte_statistics"] = byte_stats

	def save_features(self):
		# just pickle for now
		save_fn = os.path.join(self.save_dir, "{}-features.pkl".format(self.type))

		if self.append and os.path.exists(save_fn):
			current_features = pickle.load(open(save_fn,'rb'))
			for _id in self.qoe_features:
				try:
					current_features[_id]
					# don't duplicate data
				except KeyError:
					current_features[_id] = self.qoe_features[_id]
			pickle.dump(current_features, open(save_fn, 'wb'))
		else:
			pickle.dump(self.qoe_features, open(save_fn,'wb'))

	def remove_files(self, _id, link):
		# Removes stats, metadata and pcap files corresponding to this experiment. Also removes error image
		print("Removing files for type: {}, id: {}, link: {}".format(self.type, _id, link))
		call("rm {}".format(os.path.join(self.log_dir, "{}_stats_log_{}-{}-metadata.pkl".format(self.type,_id,link))),shell=True)
		call("rm {}".format(os.path.join(self.log_dir, "{}_stats_log_{}-{}-stats.csv".format(self.type,_id,link))), shell=True)
		call("rm {}".format(os.path.join(self.pcap_dir, "{}_{}.pcap".format(self.type, _id))), shell=True)
		call("rm {}".format(os.path.join(self.error_report_dir, "went_wrong_{}_{}.png".format(self.type, _id))), shell=True)

	def was_unsuccessful_experiment(self, _id):
		# check to see if there is a correspond error on this data collection
		if os.path.exists(os.path.join(self.error_report_dir, "went_wrong_{}_{}.png".format(self.type, _id))):
			return True
		return False

	def visualize_bit_transfers(self, _id):
		# Creates images showing bit transfers over time
		font = {'size'   : 6}

		matplotlib.rc('font', **font)
		ip = list(self.bytes_transfered[0].keys())[0]
		port = list(self.bytes_transfered[0][ip].keys())[0]
		n_bins = len(self.bytes_transfered[0][ip][port])
		for i in range(2):
			# up and down
			n_ips = len(self.bytes_transfered[i])
			ax = []
			try:
				fig = plt.figure(figsize=(9,13))
			except:
				# doesn't work when running via screen
				return
			for j,ip in enumerate(self.bytes_transfered[i]):
				n_ports = len(self.bytes_transfered[i][ip])
				transfer_arr = np.zeros((n_ports, n_bins))
				for k, port in enumerate(self.bytes_transfered[i][ip]):
					transfer_arr[k] = self.bytes_transfered[i][ip][port]
				ax.append(fig.add_subplot(n_ips,1,j+1))
				plt.imshow(transfer_arr)
				ax[-1].set_title("IP {} , Max {} ".format(ip,np.max(np.max(transfer_arr))))
				ax[-1].set_yticklabels(list(self.bytes_transfered[i][ip].keys()))
			if i == 0:
				plt.savefig("figures/{}_up_{}.pdf".format(self.type,_id))
			else:
				plt.savefig("figures/{}_down_{}.pdf".format(self.type,_id))
			plt.clf()
			plt.close()

	def run_no_video(self):
		pcaps = glob.glob(os.path.join(self.pcap_dir, "no_video_*.pcap"))
		print("Going through {} pcaps in no video aggregator...".format(len(pcaps)))
		for pcap in pcaps:
			_id = re.search("no_video_(.+).pcap", pcap).group(1)
			self.load_pcap(_id)
			self.populate_features("", _id)

		self.save_features()
		self.cleanup_files()		

	def run(self):
		# scroll through all the log files matching this type, make features, add to data set
		print("Running {} data aggregator.".format(self.type))
		stats_files = glob.glob(os.path.join(self.log_dir, "{}_stats_log_*.pkl".format(self.type)))
		for stats_file in stats_files:
			experiment_info = re.search(self.type + '_stats_log_(\d{10})-(.+)-metadata.pkl', stats_file)
			link = experiment_info.group(2)
			_id = experiment_info.group(1)
			if not os.path.exists(os.path.join(self.log_dir, "{}_stats_log_{}-{}-stats.csv".format(self.type,_id,link))) or self.was_unsuccessful_experiment(_id):
				self.remove_files(_id, link)
				continue
			self.load_data(link, _id)
			self.populate_features(link, _id)
		self.save_features()
		self.cleanup_files()		

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--type', action='store')
	parser.add_argument('--mode', action='store',default='run')
	args = parser.parse_args()

	da = Data_Aggregator(args.type)
	if args.mode == 'run':
		if args.type == "no_video":
			da.run_no_video()
		elif args.type in ["twitch", "youtube", "netflix"]:
			da.run()
		else:
			raise ValueError("Type {} not recognized.".format(args.type))
	elif args.mode == "visualize":
		da.visualize_bit_transfers()
	else:
		raise ValueError("Mode {} not recognized.".format(args.mode))


if __name__ == "__main__":
	main()