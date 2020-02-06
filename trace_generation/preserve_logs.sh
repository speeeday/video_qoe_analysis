# find chrome window id
wid=$(xdotool search --class Chrome | tail -n1)
# build tab key sequence to set 'persistent logs'
# HARD CODED VALUE 27, MAKE SURE THIS IS YOUR PRESERVE NETWORK LOG TAB
tabkeys=$(for i in {1..27}; do t+="Tab ";done ;echo "$t space")
# base xdotool command
cmd="xdotool windowactivate --sync $wid"
# make chrome dance :-p
#$cmd key ctrl+t
$cmd key F12
sleep 1
# open settings, move to desired option and select it
$cmd key F1 $tabkeys
sleep 1

# Close Dev Settings
$cmd key F12
sleep 1