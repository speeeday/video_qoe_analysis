TODO: add info about how to generate traces, modify BW links, set up mininet etc.

# Current Workflow
Here I am describing the current workflow to get the testbed working. These READMEs will be updated as more automation is added in.

## Starting the Topology
`python start_mininet.py --num-hosts <num_hosts> --pcap-name <pcap_name>`

This will start a topology with 2 hosts, sharing a single contested link. A pcap of the traffic over the contested link will be recorded and named as <pcap_name>

For these experiments we used Chrome running on Linux, as the Selenium browser automated had issues with video playback on Firefox.

## Generating the Data
First, make sure the HAR file will be generated for this session. Inside Developer Tools > Network tick the box to Preserve Logs.

SSL Key Logging happens using the environment variable SSLKEYLOGFILE which needs to be set in the terminal window opening up the video in a browser. This can be added to the bashrc as well. Set SSLKEYLOGFILE to the path+name of the file you would like to log the key file to.

I have written automated scripts to start the video content, provided a link to the content (make sure to put the link in quotes). For example:
`python netflix_video.py <netflix link>` will log in to netflix and play the provided video link.
This requires the credentials to be stored for the netflix username and password in the credentials folder.

After the video is finished playing we should also record the HTTP Archive for the session. Inside Developer Tools > Network > Export HAR we can save the http archive for this video session.

Current Issues
- video doesn't always start when page is visited, need to click play
- autoplay will keep running video afterwards


## Using BMV2 Switch
Eventually, we will need to replace the OVS Switch with a Virtual P4 Switch where we can test out different scheduling policies in the last mile.

This will use the BMV2 switch, found [here](https://github.com/p4lang/behavioral-model).
