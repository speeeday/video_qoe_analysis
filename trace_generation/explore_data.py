import pickle, matplotlib.pyplot as plt, os
from helpers import *
_types = ["netflix", "youtube", "twitch"]
x = {t:None for t in _types}
cdf_x = {t:None for t in _types}
for _type in _types:
	if not os.path.exists("features/{}-features.pkl".format(_type)):
		continue
	d = pickle.load(open("features/{}-features.pkl".format(_type),'rb'))
	all_stats_reports = [el for example in d.values() for el in example["stats_panel"]]
	qualities = set([el["current_optimal_res"] for el in all_stats_reports])
	buffer_healths = [float(el["buffer_health"].replace('s','')) for el in all_stats_reports]
	states = set([el["state"] for el in all_stats_reports])
	print("{} --  qualities: {}, states: {}".format(_type, qualities, states))

	ret = get_cdf_xy(buffer_healths)
	x[_type] = ret[0]
	cdf_x[_type] = ret[1]

	plt.plot(x[_type], cdf_x[_type], label=_type + "_" + str(len(all_stats_reports)))
plt.xlabel("Buffer (s)")
plt.ylabel("CDF of reports")
plt.title("Comparisons of Buffer Healths in Reports")
plt.legend()
plt.savefig("compare_buffer_health.pdf")
