from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import Controller, OVSSwitch
from mininet.log import setLogLevel, info
from mininet.cli import CLI
from mininet.link import Intf, TCIntf
from mininet.link import TCLink


from subprocess import Popen, PIPE, STDOUT, call, check_output
from threading import Thread, Lock
import queue, time, os, signal, re, numpy as np, sys, math, traceback
import zmq

import matplotlib.pyplot as plt
import matplotlib.animation as matani


from constants import *

class Proc_Out_Reader:
	def __init__(self):
		self.running = True

	def enqueue_output(self, out, queue):
		for line in iter(out.readline, b''):
			if not self.running:
				break
			queue.put(line.decode('utf-8').strip())

class SingleSwitchTopo(Topo):
	host_objects = []

	"Single switch connected to n (< 256) hosts."
	def __init__(self, n, **opts):
		# Initialize topology and default options
		Topo.__init__(self, **opts)

		switch = self.addSwitch('s1')
		switch2 = self.addSwitch('s2')
		switch3 = self.addSwitch('s3')
		switch4 = self.addSwitch('s4')
		# All units are Mb/s
		self.addLink(switch2,switch, cls=TCLink, bw=100) # BOTTLENECK LINK (all clients are limited by this link)

		self.addLink(switch2,switch3, cls=TCLink)
		self.addLink(switch2,switch4, cls=TCLink)

		
		for h in range(int(math.ceil(n/2))):
			host = self.addHost('h%d' % (h + 1))
			self.addLink(host, switch3)
			self.host_objects.append(host)

		for h in range(n-int(math.ceil(n/2))):
			host = self.addHost('h%d' % (h + int(math.ceil(n/2)) + 1))
			self.addLink(host, switch4)
			self.host_objects.append(host)

class Client_Manager:
	def __init__(self, publish_stats=False):
		self.num_hosts = 2
		self.clients = {}
		self.client_message_queue = {}
		self.client_p = {}
		self.object_lock = Lock()
		self.process_find_uids = {}
		self.threads = {}
		self.client_reports = {}
		self.t_start = time.time()

		self.periods = {
			"update_plots": {
				"f": self.update_client_stats_plots,
				"period": 2,
				"last": 0,
			},
		}

		self.publish_stats = publish_stats
		if self.publish_stats:
			self.setup_zmq_push()

		# TODO
		# start random services, random videos in those services
		# give option to randomly spawn or kill services
		# return statistics through pipe
		# plot statistics on live graph & periodically save 1) statistics, 2) plots for post processing/analysis

	def cleanup(self):
		call("sudo killall chrome", shell=True)
		call("sudo killall chromedriver", shell=True)
		call("sudo killall firefox", shell=True)
		if self.publish_stats:
			self.zmq_push_socket.close()
	
	def qoe_animation(self, i, *fargs):
		# This blocks, so would need a separate process with data communication
		# between the processes, which is annoying
		pass
		# client_id = fargs
		# row_num = client_id // self.n_rows_plot
		# col_num = client_id % self.n_rows_plot
		# ax_tuple = (row_num, col_num)
		# this_ax = self.display_objs["ax"][ax_tuple]
		# this_ax.clear()
		# report_times = [el["t"] - self.t_start for el in self.client_reports[client_id]]
		# report_bh = [el["buffer_health"] for el in self.client_reports[client_id]]
		# this_ax.scatter(report_times, report_bh)
		# this_ax.set_xlabel("Time Since Session Start (s)")
		# this_ax.set_ylabel("Buffer Health (s)")
		# plt.show()

	def get_process(self, client_id, cmd):
		return self.clients[client_id].popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)

	def setup_zmq_push(self):
		context = zmq.Context()
		self.zmq_push_socket = context.socket(zmq.PUSH)
		# self.zmq_push_socket.bind("tcp://*:{}".format(ZMQ_PORT))
		# self.zmq_push_socket.setsockopt(zmq.LINGER, 0)
		self.zmq_push_socket.connect("tcp://127.0.0.1:{}".format(ZMQ_PORT))

	def process_client_reports(self):
		try:
			self.object_lock.acquire()
			process_keys = list(self.client_message_queue.keys())
		except:
			print('Hit error in process_client_reports lock acquire -- {}'.format(sys.exc_info()))
			exit(0)
		finally:
			self.object_lock.release()
		for process_key in process_keys:
			while not self.client_message_queue[process_key].empty():
				try:
					client_report = self.client_message_queue[process_key].get(timeout=.1)
					# do something with the report
					fields = client_report.split('\t')
					if fields[0] == 'msg': 
						print("{} {}".format(process_key, client_report)) # just an informational message
					elif fields[0] == 'error': 
						print("Error in process: {}, {}".format(process_key, fields[1]))
					else:
						self.client_reports[process_key].append({
							"resolution": fields[2],
							"buffer_health": float(fields[4]),
							"state": fields[6],
							"t": time.time()
						})
				except:
					print("{} Error parsing message: {}".format(process_key, client_report))
					print(traceback.format_exc())
		if self.publish_stats:
			self.publish_latest_stats(process_keys)

	def read_client_process(self, process_names=None):
		if process_names is None:
			process_names = self.client_p.keys()
		for process_key in process_names:
			try:
				self.client_message_queue[process_key]
			except KeyError:
				try:
					self.object_lock.acquire()
					self.client_message_queue[process_key] = queue.Queue()
				except:	
					print('Hit an error acquiring lock for new packet queue key in read_client_process -- {}'.format(sys.exc_info()))
					exit(0)
				finally:
					self.object_lock.release()
			por = Proc_Out_Reader()
			self.threads[process_key]["reader"] = por
			print("Setting a reader for client: {}".format(process_key))
			por.enqueue_output(self.client_p[process_key]["process"].stdout, self.client_message_queue[process_key])

	def publish_latest_stats(self, process_keys):
		# Pushes buffer healths of all players to ZMQ socket, to be read by other interested processes
		stats_report_obj = {process_key: {"buffer": self.client_reports[process_key][-1]["buffer_health"]}
			for process_key in process_keys if len(self.client_reports[process_key]) > 0}
		if stats_report_obj == {}: return
		try:
			self.zmq_push_socket.send_pyobj(stats_report_obj)
		except zmq.ZMQError:
			print(traceback.format_exc())

	def spawn_client(self, client_id, service_type, service_link, **kwargs):
		cmd = "sudo -u ubuntu unbuffer /home/ubuntu/video_qoe_analysis/venv/bin/python {}_video.py"\
		" --link {} --id {}".format(service_type, service_link, client_id)
		self.client_p[client_id] = {
			"process": self.get_process(client_id, cmd),
			"ttl": 100,
			"t_start": time.time(),
			"cmd": cmd,
		}
		self.threads[client_id] = {
			"thread" : Thread(target=self.read_client_process, 
				kwargs={"process_names": [client_id]}, daemon=True),
			"reader": None,
		}
		self.threads[client_id]["thread"].start()
		self.process_find_uids[client_id] = "grep \"--id {}\"".format(client_id)
		self.client_reports[client_id] = []

	def update_client_stats_plots(self):
		for client in self.client_reports:
			if len(self.client_reports[client]) > 0:
				print("Client {} Report: {}".format(client, self.client_reports[client][-1]))

	def run_simple_scenario(self):
		num_hosts = self.num_hosts
		topo = SingleSwitchTopo(num_hosts)
		net = Mininet(topo = topo,
					  switch = OVSSwitch)
		net.addNAT().configDefault()
		net.start()
		try:
			time.sleep(1)
			for i in range(num_hosts):
				curr_client = net.get(topo.host_objects[i])
				self.clients[i] = curr_client
				curr_client.setIP('10.{}/8'.format(i+1))
				self.spawn_client(i, "twitch", "https://www.twitch.tv/timthetatman")
			done=False
			while not done:
				# out here go things that happen every loop
				self.process_client_reports()
				for task in self.periods:
					# heavier-weight things that should only happen every now and then
					# TODO -- add or drop clients
					if time.time() - self.periods[task]["last"] > self.periods[task]["period"]:
						self.periods[task]["f"]()
						self.periods[task]["last"] = time.time()
				time.sleep(.5)

			CLI(net)
		except:
			print(traceback.format_exc())
		finally:
			net.stop()
			self.cleanup()


def main():
	cm = Client_Manager(publish_stats=True)
	cm.run_simple_scenario()

if __name__ == "__main__":
	setLogLevel( 'error' )
	main()