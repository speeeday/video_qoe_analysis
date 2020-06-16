import pickle, matplotlib.pyplot as plt, os
from helpers import *
_types = ["netflix", "youtube", "twitch"]
x = {t:None for t in _types}
cdf_x = {t:None for t in _types}
quality_to_pixels = {}
for _type in _types:
	if not os.path.exists("features/{}-features.pkl".format(_type)):
		continue
	d = pickle.load(open("features/{}-features.pkl".format(_type),'rb'))
	all_stats_reports = [el for example in d.values() for el in example["stats_panel"]]
	qualities = set([el["current_optimal_res"] for el in all_stats_reports])
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
	buffer_healths = [float(el["buffer_health"].replace('s','')) for el in all_stats_reports]
	states = set([el["state"] for el in all_stats_reports])
	print("{} --  qualities: {}, states: {}".format(_type, qualities, states))

	ret = get_cdf_xy(buffer_healths)
	x[_type] = ret[0]
	cdf_x[_type] = ret[1]

	plt.plot(x[_type], cdf_x[_type], label=_type + "_" + str(len(all_stats_reports)))
	print(sorted(quality_to_pixels[_type].items(), key= lambda el : el[1]))
plt.xlabel("Buffer (s)")
plt.ylabel("CDF of reports")
plt.title("Comparisons of Buffer Healths in Reports")
plt.legend()
plt.savefig("compare_buffer_health.pdf")
