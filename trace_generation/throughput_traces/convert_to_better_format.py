import numpy as np, matplotlib.pyplot as plt,csv, os
def create_throughput_trace(granularity=1): # granularity is in seconds
	means = []
	raw_trace_dir = "./raw_pensieve_traces"
	save_dir = "./traces"
	mtu = 1500 # bytes
	for trace_file in os.listdir(raw_trace_dir):
		# data is a list of milliseconds
		with open(os.path.join(raw_trace_dir, trace_file), 'r') as f:
			data = [float(el.strip()) for el in f.readlines()]
		last_granularity = data[-1] / (granularity * 1000) + 1
		all_intervals = np.zeros((int(last_granularity)))
		for el in data:
			all_intervals[int(el/(granularity*1000))] += mtu
		# the last interval spills over, just keep it the same
		all_intervals[-1] = all_intervals[-2] 
		all_intervals = [el/7 for el in all_intervals]
		means.append(np.mean(all_intervals) * 8 / 1e6)
		# plt.plot(list(range(len(all_intervals))), all_intervals/1e6)
		# plt.xlabel("Time ({}-s intervals)".format(granularity))
		# plt.ylabel("BW (MBps)")
		# plt.show()
		all_intervals = [(i*granularity,el) for i,el in enumerate(all_intervals)]
		# save the data
		with open(os.path.join(save_dir,trace_file + ".csv"), 'w') as f:
			csvw = csv.writer(f)
			csvw.writerows(all_intervals)
	print("Means: {} , Mean of Means: {}, Median of means: {}".format(
		means, np.mean(means), np.median(means)))



if __name__ == "__main__":
	create_throughput_trace(5)