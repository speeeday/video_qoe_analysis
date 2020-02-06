#!/bin/bash
if [ "$1" == "" ]
then
	echo "Usage : ./run_youtube.sh [link of youtube video]"
else
	# Call wireshark
	# filters should be added here to remove typical traffic seen on the network
	# more filters = less post-processing
	sudo tshark -i eth0 -f "not (port ssh or port ntp)" -w pcaps/youtube_experiment.pcap &
	# Call the python program
	source ../venv/bin/activate
	python youtube_video.py --link "$1"

	sudo killall tshark
fi