"""
Example to create a Mininet topology and connect it to the internet via NAT
"""


from mininet.cli import CLI
from mininet.log import lg, info
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
if __name__ == '__main__':
	lg.setLogLevel( 'info')
	myTopo = StaticTopo()
	net = Mininet( topo=myTopo, link=TCLink )
	h1 = net.get('h1')
	# Add NAT connectivity
	net.addNAT().configDefault()
	net.start()
	h1.cmd("dig google.com")
	info( "*** Hosts are running and should have internet connectivity\n" )
	info( "*** Type 'exit' or control-D to shut down network\n" )
	CLI( net )
	# Shut down NAT
	net.stop()