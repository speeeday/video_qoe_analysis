TODO: add info about how to generate traces, modify BW links, set up mininet etc.

## Starting the Topology
`python start_mininet.py --num-hosts <num_hosts> --pcap-name <pcap_name> --link <video_link>`

Current Issues
- video doesn't always start when page is visited, need to click play
- autoplay will keep running video afterwards


## Using BMV2 Switch
Eventually, we will need to replace the OVS Switch with a Virtual P4 Switch where we can test out different scheduling policies in the last mile.

This will use the BMV2 switch, found [here](https://github.com/p4lang/behavioral-model).