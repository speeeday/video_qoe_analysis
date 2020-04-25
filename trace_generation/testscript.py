import os
import sys
import subprocess
from subprocess import Popen, PIPE
import shlex
import getpass

print ("This script was called by: " + getpass.getuser())
proc = subprocess.call("bash bscript.sh",shell=True)
#proc = subprocess.call('',env=new_env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
#subprocess.call("bscript.sh",shell=True)
#subprocess.call(shlex.split('bash bscript.sh'))
#outfile = open('tttttest','w')
#args = shlex.split("su -l nillin")
print ("This script was now called by: " + getpass.getuser())
#Popen(args)
#args = shlex.split("ls -a")
#Popen(args,stdout = outfile)
#print ("switch user success")
