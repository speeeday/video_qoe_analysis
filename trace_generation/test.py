import time, re
from subprocess import check_output
# Mininet libraries
from mininet.net import Mininet
from mininet.link import Intf
from mininet.log import setLogLevel, info
from mininet.topo import Topo
from mininet.link import TCLink

class StaticTopo(Topo):
    "Simple topo with 2 hosts"
    def build(self):
        switch1 = self.addSwitch('s1')

        "iperf server host"
        host1 = self.addHost('h1')
        # this link is not the bottleneck
        self.addLink(host1, switch1, bw = 1000) 

        "iperf client host"
        host2 = self.addHost('h2')
        self.addLink(host2, switch1, bw = 1000)
       

# set up the mininet
myTopo = StaticTopo()
net = Mininet( topo=myTopo, link=TCLink )
switch = net.switches[0]
switch_intf = Intf('ens5',node=switch)
#net.addNAT().configDefault()
net.start()

h1 = net.get('h1')
intf = h1.intf()

# call the data collection script
cmd = "sudo -H -u ubuntu bash -c \'~/video_qoe_analysis/venv/bin/python netflix_video.py --link https://www.netflix.com/watch/80046429 --id 4\'"
cmd = "dig google.com"
print("Starting script...")
x = h1.cmd(cmd,printPid=True).strip()
print(x)
pid = int(re.search("\[(.+)\] (.+)", x).group(2))
print("Process is running with PID: {}".format(pid))
still_going = True
while still_going:
	print("process is still running...")
	time.sleep(1)
	try:
		still_going = check_output("ps -p {}".format(pid), shell=True)
	except:
		# fails when process ends
		break
print("Done!")
net.stop()
