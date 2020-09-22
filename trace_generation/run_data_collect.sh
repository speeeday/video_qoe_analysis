#!/bin/bash
if [ "$1" == "" ]
then echo "Usage : ./run_data_collect.sh [name of service] [link of stream]"
else
	# Call wireshark
	# filters should be added here to remove typical traffic seen on the network
	# more filters = less post-processing

	time_experiment=$(date +"%s")
	pcap_fname="pcaps/$1_${time_experiment}.pcap"
	sudo tshark -Q -i ens5 -f "not (port ssh or port ntp)" -w $pcap_fname &
	# Call the program
	source ../venv/bin/activate
	if [ "$1" == "youtube" ]
	then
		~/video_qoe_analysis/venv/bin/python youtube_video.py --link "$2" --id $time_experiment --mode run
	elif [ "$1" == "netflix" ]
	then
		~/video_qoe_analysis/venv/bin/python netflix_video.py --link "$2" --id $time_experiment
	elif [ "$1" == "twitch" ]
	then
		~/video_qoe_analysis/venv/bin/python twitch_video.py --link "$2" --id $time_experiment
	elif [ "$1" == "no_video" ]
	then
		~/video_qoe_analysis/venv/bin/python no_video.py
	else
		echo "Name of service not valid."
	fi
	sudo killall tshark
	sudo chmod 666 $pcap_fname
	# # get bitrates # This is making youtube block me because it requests lots of times, only use it when you need to
	# if [ "$1" == "youtube" ]
	# then
	# 	python youtube_video.py --link "$2" --id $time_experiment --mode get_bitrate
	# fi
fi