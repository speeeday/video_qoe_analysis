#!/bin/sh

ssl_key_dir="/home/sj/research/video_qoe_analysis/trace_generation/ssl_keys/"
pcap_name="s2-eth2"
fixed="-ssl_key.log"
video_url='www.youtube.com/watch?v=lZ3bPUKo5zc'
export SSLKEYLOGFILE=${ssl_key_dir}${pcap_name}${fixed}
echo "$SSLKEYLOGFILE" >> envtext.txt
sudo -u nillin whoami>> envtext.txt
echo $video_url >> videolog.txt
sudo -u nillin python youtube_new.py >> videolog.txt
echo "end of script" >> envtext.txt

