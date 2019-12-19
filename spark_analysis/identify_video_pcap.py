import pyshark
from time import sleep

def get_capture_count(pcap_name):
    p = pyshark.FileCapture(pcap_name, keep_packets=False)
    
    count = []
    def counter(*args):
        count.append(args[0])

    p.apply_on_packets(counter, timeout=100000)

    return len(count)

pcap_name = sys.argv[1]

num_pkts = get_capture_count(pcap_name)

print "Number of packets: {}".format(num_pkts)

cap = pyshark.FileCapture(pcap_name)
for i in range(num_pkts):
    pkt = cap[i]
    print pkt
    sleep(0.1)
