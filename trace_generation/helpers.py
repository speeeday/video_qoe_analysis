from constants import *
import numpy as np, os, pickle
from bisect import bisect_left

def get_difference(set1, set2):
	"""Gets set1 - set2."""
	set1 = set(set1); set2 = set(set2)
	return list(set1.difference(set2))
def get_asn_dist(asns):
	# ASN_LIST -> %
	asn_dist = np.zeros(len(ASN_LIST))
	for asn in asns:
		try:
			asn_dist[ASN_LIST[asn]] += 1
		except KeyError:
			asn_dist[ASN_LIST["OTHER"]] += 1
	assert np.sum(asn_dist) > 0

	return asn_dist / np.sum(asn_dist) 

def get_ip_likelihood(ip_list, _type, modify=False):

	ip_likelihood = 0
	if os.path.exists(os.path.join(METADATA_DIR, KNOWN_IP_LIST_FN)):
		known_ips = pickle.load(open(os.path.join(METADATA_DIR, KNOWN_IP_LIST_FN),'rb'))
	else:
		known_ips = {"_types": ["twitch","youtube","netflix"], "data": {"twitch": [], "youtube": [], "netflix": []}}
	_types = known_ips["_types"]
	# TODO update this as model gets more complicated
	ip_likelihood = [len(set(ip_list) & set(known_ips["data"][_type])) / len(set(ip_list)) for _type in _types]

	if _type == "no_video":
		# no_video ips are too numerous to track
		return ip_likelihood

	
	if modify:
		# These IPs were communicated with when accessing this service, 
		# add them to the data structure
		# TODO - incorporate some sort of frequency, staleness, likelihood thing here
		for ip in ip_list: # maybe change this to /24, although these could be v6 IP's, and there's no /24 analogy there
			known_ips["data"][_type].append(ip)
		known_ips["data"][_type] = list(set(known_ips["data"][self.type]))
		pickle.dump(known_ips, open(os.path.join(self.metadata_dir, self.known_ip_list_fn),'wb'))

	return ip_likelihood

class discrete_cdf:
	# Got from https://tinyurl.com/y6dlvbsb
	def __init__(self, data,weighted=False):
		self.weighted=weighted
		if weighted:
			# assume data is tuple (value, count of value)
			self._data = [el[0] for el in data]
			self._counts = [el[1] for el in data]
			self._data_len = float(np.sum(self._counts)) # "length" is number of everything
		else:
			self._data = data
			self._data_len = float(len(data))

	def __call__(self, point):
		if self.weighted:
			return np.sum(self._counts[:bisect_left(self._data, point)]) / self._data_len
		else:
			return (len(self._data[:bisect_left(self._data, point)]) /
				self._data_len)

def get_cdf_xy(data, logx=False, logy=False, n_points = 500, weighted=False):
	"""Returns x, cdf for your data on either log-lin or lin-lin plot."""

	# sort it
	if weighted:
		data.sort(key = lambda val : val[0]) # sort by the value, not the weight of the value
	else:
		data.sort()

	if logx:
		if weighted:
			if data[0][0] <= 0:
				log_low = -1; 
			else:
				log_low = np.floor(np.log10(data[0][0]))
			log_high = np.ceil(np.log10(data[-1][0]))
		else:
			if data[0] <= 0: # check for bad things before you pass them to log
				log_low = -1
			else:
				log_low = np.floor(np.log10(data[0]))
			log_high = np.ceil(np.log10(data[-1]))
		x = np.logspace(log_low,log_high,num=n_points)
	elif logy:
		# Do an inverted log scale on the y axis to get an effect like
		# .9, .99, .999, etcc
		log_low = -5
		log_high = 0
		x = np.linspace(data[0], data[-1], num=n_points)
	else:
		if weighted:
			x = np.linspace(data[0][0], data[-1][0], num=n_points)
		else:
			x = np.linspace(data[0], data[-1],num=n_points)

	# Generate the CDF
	cdf_data_obj = discrete_cdf(data, weighted=weighted)
	cdf_data = [cdf_data_obj(point) for point in x]

	return [x, cdf_data]