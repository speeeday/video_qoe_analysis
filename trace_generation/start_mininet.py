#!/usr/bin/env python2

# Copyright 2013-present Barefoot Networks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import Controller, OVSSwitch
from mininet.log import setLogLevel, info
from mininet.cli import CLI
from mininet.link import Intf


import argparse
from time import sleep

parser = argparse.ArgumentParser(description='Mininet demo')

parser.add_argument('--num-hosts', help='Number of hosts to connect to switch',
                    type=int, action="store", default=2)

args = parser.parse_args()

class SingleSwitchTopo(Topo):
    "Single switch connected to n (< 256) hosts."
    def __init__(self, n, **opts):
        # Initialize topology and default options
        Topo.__init__(self, **opts)

        switch = self.addSwitch('s1')
        switch2 = self.addSwitch('s2')
        
        for h in xrange(n):
            host = self.addHost('h%d' % (h + 1))
            self.addLink(host, switch2)
            
        self.addLink(switch2,switch) # BOTTLENECK LINK (all clients are limited by this link)

            
def main():

    num_hosts = args.num_hosts

    topo = SingleSwitchTopo(num_hosts)

    net = Mininet(topo = topo,
                  switch = OVSSwitch)

    net.addNAT().configDefault()
        
    net.start()

    sleep(1)

    print "Ready !"

    CLI( net )
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    main()
