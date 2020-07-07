#!/bin/sh

#ssl_key_dir="/home/sj/research/video_qoe_analysis/trace_generation/results/$1/"
fixed="ssl_key$2.log"
#export SSLKEYLOGFILE=${ssl_key_dir}${fixed}
#echo "$SSLKEYLOGFILE" >> /home/sj/research/video_qoe_analysis/trace_generation/results/$1/envtext$2.txt
#sudo -u sj whoami>> /home/sj/research/video_qoe_analysis/trace_generation/results/$1/envtext$2.txt
sudo -u sj python2.7 run_netflix_client_with_stats.py --id $1 --num $2 >> /home/sj/research/video_qoe_analysis/trace_generation/results/$1/videolog$2.txt
#echo "end of script" >> /home/sj/research/video_qoe_analysis/trace_generation/results/$1/envtext$2.txt
