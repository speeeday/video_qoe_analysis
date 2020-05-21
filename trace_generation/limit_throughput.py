from constants import *
import numpy as np, csv, os, time
from subprocess import call, check_output

def limit_throughput(file):
	traces_dir = TRACES_DIR
	if file is None:
		# choose a random file to use
		trace_file = np.random.choice(os.listdir(traces_dir))
	#print("Loading trace: {}".format(trace_file))
	with open(os.path.join(traces_dir,trace_file), 'r') as f:
		csvr = csv.reader(f)
		bandwidth_trace = [[float(el[0]), float(el[1])] for el in csvr]
	bandwidth_trace = np.array(bandwidth_trace)
	total_time = bandwidth_trace[-1,0]
	t_start = time.time()
	last_bw_restriction = None
	time.sleep(4) # give it a little boost for loading the page
	while True:
		interval = np.where(int(time.time()-t_start) % total_time >= bandwidth_trace[:,0])[0][-1]
		bw_restriction = bandwidth_trace[interval][1] # In bytes/sec, convert to Kbps
		bw_restriction = int(bw_restriction * 8 / 1000)
		if bw_restriction != last_bw_restriction:
			# set the new bw restriction
			#print("Setting bandwidth restriction to {} kbps".format(bw_restriction))
			call("tcset ens5 --overwrite --rate {}Kbps --direction incoming --src-port 443".format(bw_restriction), shell=True)
			# outgoing doesn't really matter, since relatively little data is exiting the network
			#call("tcset ens5 --rate {}Kbps --direction outgoing".format(bw_restriction), shell=True)
			#print(check_output("tcshow ens5", shell=True))
			last_bw_restriction=bw_restriction
		time.sleep(.3)

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', action='store', default=None)
	args = parser.parse_args()

	limit_throughput(args.file)

if __name__ == "__main__":
	main()