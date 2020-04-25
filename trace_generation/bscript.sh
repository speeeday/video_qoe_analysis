#!/bin/sh

echo "$HOME"
export HOME=$HOME/video_qoe_analysis

echo "$HOME" >> envtext.txt
sudo -u nillin whoami>> envtext.txt
