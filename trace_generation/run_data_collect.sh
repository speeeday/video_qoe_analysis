#!/bin/bash
if [ "$1" == ""  -o "$2" == "" ]
then echo "Usage : ./run_data_collect.sh [name of service] [link of stream]"
else
	# Call wireshark
	# filters should be added here to remove typical traffic seen on the network
	# more filters = less post-processing
	echo "pcaps/{$1}_experiment.pcap"
	sudo tshark -i eth0 -f "not (port ssh or port ntp)" -w "pcaps/$1_experiment.pcap" &
	# Call the program
	source ../venv/bin/activate
	if [ "$1" == "youtube" ]
	then
		python youtube_video.py --link "$2"
	elif [ "$1" == "netflix" ]
	then
		python netflix_video.py --link "$2"
	elif [ "$1" == "twitch" ]
	then
		python twitch_video.py --link "$2"
	else
		echo "Name of service not valid."
	fi
	sudo killall tshark
fi