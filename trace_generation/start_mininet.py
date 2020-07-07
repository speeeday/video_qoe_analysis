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
from mininet.link import Intf, TCIntf
from mininet.link import TCLink

import argparse
import os
from time import sleep, time
from subprocess import call
import math

parser = argparse.ArgumentParser(description='Mininet demo')

parser.add_argument('--num-hosts', help='Number of hosts to connect to switch',
                    type=int, action="store", default=1)

parser.add_argument('--id', help='Name of PCAP file to save',
                    type=str, action="store", default='temp')

args = parser.parse_args()

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

        self.addLink(switch2,switch, cls=TCLink, bw=10) # BOTTLENECK LINK (all clients are limited by this link)

        self.addLink(switch2,switch3, cls=TCLink)#, bw=6) # BOTTLENECK LINK (all clients are limited by this link)
        self.addLink(switch2,switch4, cls=TCLink)#, bw=6) # BOTTLENECK LINK (all clients are limited by this link)

        
        for h in xrange(int(math.ceil(n/2))):
            host = self.addHost('h%d' % (h + 1))
            self.addLink(host, switch3)
            self.host_objects.append(host)

        for h in xrange(n-int(math.ceil(n/2))):
            host = self.addHost('h%d' % (h + int(math.ceil(n/2)) + 1))
            self.addLink(host, switch4)
            self.host_objects.append(host)

            
        # BW is in Mbits
#        self.addLink(switch2,switch, cls=TCLink, bw=12) # BOTTLENECK LINK (all clients are limited by this link)

            
def main():

    logfile_dir = 'results/' + args.id
    
    if not os.path.exists(logfile_dir):
        call("mkdir -p {}".format(logfile_dir), shell=True)
    else:
        print("ID: '{}' already exists.")
        os.sys.exit()
    
    outfile = logfile_dir + "/" + args.id
    
    num_hosts = args.num_hosts

    topo = SingleSwitchTopo(num_hosts)

    
    net = Mininet(topo = topo,
                  switch = OVSSwitch)

    net.addNAT().configDefault()
        
    net.start()
    
    sleep(1)


#    raw_input()
    os.system('sudo touch {}'.format(outfile+".pcapng"))
#    os.system('sudo touch {}'.format(outfile+"-downlink.pcapng"))
    
    os.system('sudo dumpcap -i s1-eth1 -w {} &'.format(outfile+".pcapng"))
#    os.system('sudo dumpcap -i s2-eth1 -w {} &'.format(outfile+"-downlink.pcapng"))
    sleep(3)

    os.system('sudo chown sj:sj /home/sj/research/video_qoe_analysis/trace_generation/results/{}/'.format(args.id))
    sleep(1)
    
    ### EXPERIMENT CODE START ###

    # this is where you can modify how you want the clients to access the videos, for example if you want to induce a delay between when 1 client starts the stream versus the other, OR if you want to change the streaming services between different clients that can all be modified here
    


    
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
#    client.cmd('xterm &')
#    client.cmd('xterm &')
#    client2.cmd('xterm &')

    for i in range(num_hosts):
        curr_client = net.get(topo.host_objects[i])
        curr_client.cmd('bash /home/sj/research/video_qoe_analysis/trace_generation/run_netflix_client_with_stats.sh {} {} &'.format(args.id, i+1), shell=True)

#    client1 = net.get(topo.host_objects[0])
#    client2 = net.get(topo.host_objects[1])
#    client3 = net.get(topo.host_objects[2])
#    client4 = net.get(topo.host_objects[3])

#    video_url = args.link

#    ssl_key_dir = '/home/sj/research/video_qoe_analysis/trace_generation/results/' + args.id + '/'


#    client1.cmd('bash /home/sj/research/video_qoe_analysis/trace_generation/run_netflix.sh {} {} &'.format(args.id, 1), shell=True)
#    client2.cmd('bash /home/sj/research/video_qoe_analysis/trace_generation/run_netflix.sh {} {} &'.format(args.id, 2), shell=True)
#    client3.cmd('bash /home/sj/research/video_qoe_analysis/trace_generation/run_netflix.sh {} {} &'.format(args.id, 3), shell=True)
#    client4.cmd('bash /home/sj/research/video_qoe_analysis/trace_generation/run_netflix.sh {} {} &'.format(args.id, 4), shell=True)

   # rename the ssl key log
#    client.cmd('mv /home/sj/research/video_qoe_analysis/trace_\
#generation/ssl_keys/curr-ssl_key.log ' + ssl_key_dir + args.id + '-ssl_key.log')
#    client2.cmd('mv /home/sj/research/video_qoe_analysis/trace_\
#generation/ssl_keys/curr-ssl_key.log ' + ssl_key_dir + args.id + '-ssl_key.log')
    
    ### EXPERIMENT CODE END   ###
    
    print "Done !"
    # Hang on the CLI before we exit manually

    CLI( net )
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    main()
