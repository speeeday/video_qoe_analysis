import matplotlib
#matplotlib.use('Agg')
import pickle, matplotlib.pyplot as plt, os
from helpers import *
from constants import *

# Script used to look at statistics of streams (what typical buffer healths are, chunk sizes, etc..)

_types = ["netflix", "youtube", "twitch"]
_types = ["twitch"]
x = {t:None for t in _types}
cdf_x = {t:None for t in _types}
quality_to_pixels = {}

buffer_healths = {}
buffer_health_deltas = {}
bitrates = {}
for _type in _types:
	if not os.path.exists("features/{}-features.pkl".format(_type)):
		continue
	d = pickle.load(open("features/{}-features.pkl".format(_type),'rb'))
	all_stats_reports = [el for example in d.values() for el in example["stats_panel"]]
	qualities = set([el["current_optimal_res"] for el in all_stats_reports])
	if _type == "twitch":
		bitrates[_type] = [float(el["bitrate"])*1000 for el in all_stats_reports]
	quality_to_pixels[_type] = {}
	for quality in qualities:
		if _type in ['netflix', 'twitch']:
			_x,_y = quality.split('x')
			quality_to_pixels[_type][quality] = int(_x) * int(_y)
		else:
			if quality == "0x0":
				quality_to_pixels[_type][quality] = 0
			else:
				experienced,optimal = quality.split("/")
				experienced = [int(el) for el in experienced.split("@")[0].split("x")]
				quality_to_pixels[_type][quality] = experienced[0] * experienced[1]
	buffer_healths_this_type = [float(el["buffer_health"].replace('s','')) for el in all_stats_reports]
	states = set([el["state"] for el in all_stats_reports])
	# print("{} --  qualities: {}, states: {}".format(_type, qualities, states))

	ret = get_cdf_xy(buffer_healths_this_type)
	buffer_healths[_type] = ret

	# print(sorted(quality_to_pixels[_type].items(), key= lambda el : el[1]))

	# Get buffer health deltas
	deltas = []
	for example in d.values():
		last_bh = float(example["stats_panel"][0]["buffer_health"].replace('s',''))
		for el in example["stats_panel"][1:]:
			this_bh = float(el["buffer_health"].replace('s',''))
			if this_bh > last_bh:
				deltas.append(this_bh-last_bh)
			last_bh = this_bh
	buffer_health_deltas[_type] = deltas

# Plot bitrates
for _type in bitrates:
	x,cdf_x = get_cdf_xy(bitrates[_type])
	plt.plot(x,cdf_x,label=_type)
plt.xlabel("Bitrate (b/s)")
plt.ylabel("CDF of Reports")
plt.legend()
plt.savefig("figures/bitrates.pdf")

# Plot buffer health deltas
for _type in buffer_health_deltas:
	plt.hist(buffer_health_deltas[_type],bins=50)
	plt.savefig("figures/BHD-{}.pdf".format(_type))
	plt.clf()
	plt.close()

# Plot buffer healths
n_classes = 10
percentiles = np.linspace(0,.99,n_classes+1)[1:]
for _type in buffer_healths:
	x,cdf_x = buffer_healths[_type]
	plt.plot(buffer_healths[_type][0], buffer_healths[_type][1], label=_type)
	print("Type: {} - max: {}".format(_type, np.max(buffer_healths[_type][0])))
	print("Mean: {}".format(np.mean(buffer_healths[_type])))
	print("Random guessing MAE: {}".format(np.mean(np.abs(np.mean(buffer_healths[_type]) - buffer_healths[_type]))))
	# Look at percentiles of interest, to determine 
	# splitting points for equal amounts of data in each class
	for p_ile in percentiles:
		stat = x[[i for i,val in enumerate(cdf_x) if val >= p_ile][0]],
		print("{}th %ile -- {}".format(int(p_ile*100), stat))
plt.xlabel("Buffer (s)")
plt.ylabel("CDF of reports")
plt.title("Comparisons of Buffer Healths in Reports")
plt.legend()
plt.savefig("figures/BH.pdf")


# Look at buffer features
buffer_features = pickle.load(open("features/buffer_regression-val.pkl",'rb'))
examples = buffer_features["X"]
by_channel = []
for i in range(TOTAL_N_CHANNELS):
	by_channel.append([])
	for el in examples:
		flattened_el = np.sum(el[:,:,i],axis=1).flatten()
		for num in flattened_el:
			by_channel[i].append(num)
f,ax = plt.subplots(3,3)
for i in range(TOTAL_N_CHANNELS):
	row_i = i % 3
	col_i = i // 3
	x,cdf_x = get_cdf_xy(by_channel[i])
	ax[row_i,col_i].semilogx(x,cdf_x)
	ax[row_i,col_i].set_title("Channel {}".format(i))
plt.show()
