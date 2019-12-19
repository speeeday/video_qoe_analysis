#!/usr/bin/python
import os
import sys
from ip_protocols import *
from fields import fields

def proto(protocol_name):
    p = protocol_name.lower()
    if p not in ip_protocols:
        print "Unknown Protocol: {}".format(p)
        sys.exit()
    return ip_protocols[p]

# argv[1] - pcap_name
# argv[2] - outfile_name

pcap_name = sys.argv[1]
try:
    outfile_name = sys.argv[2]
except:
    outfile_name = ''    

# tshark -T fields -n -r gilberto.pcap -E separator='|' -e frame.number -e ip.src_host -e ip.dst_host > test

cmd = ''
cmd += "tshark -T fields -n -r {} -E separator='|'".format(pcap_name)
for f in fields:
    cmd += ' -e {}'.format(f)

if outfile_name != '':
    cmd += ' > {}'.format(outfile_name)
    
print "Executing '{}'".format(cmd)

# call tshark
os.system(cmd + '\n')
