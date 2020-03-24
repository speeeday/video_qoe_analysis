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
import os
from time import sleep

parser = argparse.ArgumentParser(description='Mininet demo')

parser.add_argument('--num-hosts', help='Number of hosts to connect to switch',
                    type=int, action="store", default=2)

parser.add_argument('--pcap-name', help='Name of PCAP file to save',
                    type=str, action="store", default='pcaps/s2-eth2.pcapng')

parser.add_argument('--link', help='Name of video link to use',
                    type=str, action="store", default='https://www.youtube.com/watch?v=NKaA0IPcD_Q')

args = parser.parse_args()

class SingleSwitchTopo(Topo):
    host_objects = []

    "Single switch connected to n (< 256) hosts."
    def __init__(self, n, **opts):
        # Initialize topology and default options
        Topo.__init__(self, **opts)

        switch = self.addSwitch('s1')
        switch2 = self.addSwitch('s2')
        
        for h in range(n):
            host = self.addHost('h%d' % (h + 1))
            self.addLink(host, switch2)
            self.host_objects.append(host)
            
        self.addLink(switch2,switch) # BOTTLENECK LINK (all clients are limited by this link)

            
def main():

    outfile = args.pcap_name
    
    num_hosts = args.num_hosts

    topo = SingleSwitchTopo(num_hosts)

    
    net = Mininet(topo = topo,
                  switch = OVSSwitch)

    net.addNAT().configDefault()
        
    net.start()
    
    sleep(1)


#    raw_input()
    os.system('sudo dumpcap -i s1-eth1 -w {} &'.format(outfile))
    sleep(3)
    
    ### EXPERIMENT CODE START ###

    # this is where you can modify how you want the clients to access the videos, for example if you want to induce a delay between when 1 client starts the stream versus the other, OR if you want to change the streaming services between different clients that can all be modified here
    
    client = net.get(topo.host_objects[0])
    print(client.cmd("dig google.com"))

    video_url = args.link

    ssl_key_dir = '/home/sj/research/video_qoe_analysis/trace_generation/ssl_keys/'
    
#    client.cmd('export SSLKEYLOGFILE=' + ssl_key_dir + args.pcap_name.split('/')[1].split('.')[0] + '-ssl_key.log')

#    client.cmd('su - sj')
#    client.cmd('firefox {} &'.format(video_url))
#    client.cmd('chromium-browser --no-sandbox --ignore-certificate-errors --user-data-dir=/tmp/nonexistent$(date +%s%N) {} &'.format(video_url))    

   # current experiment:
   # SSLKEYLOGFILE is in .bashrc
   # sudo - sj
   # firefox <url>
   # enable HAR (Developer Tools > Web Console > Persist Logs + Clear)
   # play video
   # close window once video is finished
   # rename ssl key based on pcap name


   # open a window on h1 to run experiment manually (until sslkeylogfile can be automated)
    client.cmd('xterm')
   
    ### EXPERIMENT CODE END   ###
    
    print("Done !")
    # Hang on the CLI before we exit manually

    CLI( net )
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    main()
