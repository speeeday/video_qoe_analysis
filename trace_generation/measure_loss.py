import pyshark
from time import sleep
import sys
import os
from subprocess import call
from scipy.stats.stats import pearsonr, variation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import itertools
import pandas as pd

def crosscorr(x, y, lag=2, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    datax = pd.Series(x)
    datay = pd.Series(y)
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))    


parser = argparse.ArgumentParser(description='Mininet demo')

parser.add_argument('--num-hosts', help='Number of hosts to connect to switch',
                    type=int, action="store", default=3)

parser.add_argument('--delta', help='Time Length of each window',
                    type=int, action="store", default=15)

parser.add_argument('--window-size', help='Number of windows before to consider when calculating correlation (0 is all before)',
                    type=int, action="store", default=0)

parser.add_argument('--pcap', help='Name of PCAP',
                    type=str, action="store", required=True)

args = parser.parse_args()

num_hosts = args.num_hosts

window_size = args.window_size

# stats, timestamp, pcap

#print "Input the time in PCAP for video request:"
#init_time = float(raw_input())

#lines = [l.split(',') for l in open(sys.argv[1], 'r').read().strip('\n').split('\n')]

#time_init = float(open(sys.argv[2], 'r').read())

first = True

#stats = {}
#times = []
#for l in lines:
#    if first:
#        first = False
#        continue
    #time, bitrate, buffer_health, throughput
#    print "{}\t{}\t{}\t{}".format(float(l[4])-time_init,l[3],l[5],l[0])
#    time = float(l[4])-time_init
#    stats[time] = "\t\t\t\t{}\t{}\t{}".format(l[3],l[5],l[0])
#    times.append(time)

def closest(t):
    absolute_difference_function = lambda list_value : abs(list_value - t)
    closest_value = min(times, key=absolute_difference_function)
    return closest_value

    
pcap_name = args.pcap
path = "/".join(pcap_name.split("/")[:-1])


all_records = []

print "Processing {}".format(pcap_name)

server_ips = []

# list hostnames
call("tshark -Y \"dns.flags.response == 1\" -T fields -n -r {} -E separator='/t' -e frame.time_relative -e ip.proto -e ip.src_host -e tcp.srcport -e ip.dst_host -e tcp.dstport -e frame.len -e dns.qry.name -e dns.a > ./temp".format(pcap_name), shell=True)

for l in open('./temp').read().strip('\n').split('\n'):
    p = l.strip('\t').split('\t')
    try:
        hostname = p[7]
        if 'netflix' in hostname or 'nflx' in hostname:
            ips = p[8].split(',')
            server_ips += ips
    except:
        continue
            
call("tshark -T fields -n -r {} -E separator='/t' -e frame.time_relative -e ip.proto -e ip.src_host -e tcp.srcport -e ip.dst_host -e tcp.dstport -e frame.len -e tcp.flags -e tcp.seq_raw -e tcp.ack_raw -e tcp.analysis.retransmission > ./temp".format(pcap_name), shell=True)

for l in open('./temp').read().strip('\n').split('\n'):
    p = l.strip('\t').split('\t')
    cp = {}
    try:
        cp['time'] = float(p[0])
    except:
        cp['time'] = None
    try:
        cp['protocol'] = int(p[1])
    except:
        cp['protocol'] = None
    try:
        cp['src_ip'] = p[2]
        cp['src_port'] = p[3]
        cp['dst_ip'] = p[4]
        cp['dst_port'] = p[5]
    except:
        continue
    try:
        cp['size'] = int(p[6])
    except:
        cp['size'] = None
    try:
        cp['tcp_flags'] = int(p[7],16)
    except:
        cp['tcp_flags'] = None
    try:
        cp['seq_num'] = int(p[8])
    except:
        cp['seq_num'] = None
    try:
        cp['ack_num'] = int(p[9])
    except:
        cp['ack_num'] = None
    try:
        # TODO: FIX THIS X
        flg = int(p[10])
        if flg == 1:
            cp['retransmission'] = True
        else:
            cp['retransmission'] = False            
    except:
        cp['retransmission'] = False
    all_records.append(cp)

# List hostnames requested in DNS requests

# Retrieve IPs for the video stream that is in the DNS answers

bytes_sent = {}
cum_bytes_sent = {}
losses = {}
cum_losses = {}
binarized_losses = {}
loss_rates = {}
variation_loss_rates = {}
variation_loss_rates_xs = {}

client_acks = {}

def get_ack_times(ip):
    

def get_loss_per_window(ip):

    filtered_records = []

    # Currently only looking at arrival times
    for r in all_records:
        if r['src_ip'] in server_ips and r['protocol'] == 6 and r['dst_ip'] == ip:
            filtered_records.append(r)

    #init_time = 98.349467054
    #init_time = 60.436651887
#    for ir in range(0, len(filtered_records)):
#        r = filtered_records[ir]
#        if r['retransmission']:
#            t = r['time'] - init_time
#            if t > 0:
#                print str(t) + "\t" + stats[closest(t)]

    # WINDOW SIZE
    BW_DELTA = args.delta * 1.0
    packet_groups = []
    curr_group = []
    group_num = 1
    for ir in range(0,len(filtered_records)):
        r = filtered_records[ir]
        if (r['time'] < BW_DELTA*group_num):
            curr_group.append(r)
        else:
            packet_groups.append(curr_group)
            curr_group = []
            group_num += 1
            curr_group.append(r)

#    print "Number of Groups: {}".format(len(packet_groups))

    win_num = 0
    losses = []
    loss_rates = []
    bytes_sent = []
    for b in packet_groups:
        win_num += 1
        bs = 0
        rc = 0
        c = 0
        for p in b:
            c += 1
            bs += p['size']
            if p['retransmission']:
                rc += 1
        if len(b) > 1:
            loss_rates.append((rc*1.0)/(c*1.0))
            losses.append(rc)
            bytes_sent.append(bs)
            ### NOTE: can either do absolute # of losses or loss ratio within a certain window
            ### also there is inter -loss time
    return losses, loss_rates, bytes_sent

for i in range(1,num_hosts+1):
    aks = get_ack_times('10.0.0.{}'.format(i))
    ls, lrs, bss = get_loss_per_window('10.0.0.{}'.format(i))
    losses['H{}'.format(i)] = ls
    binarized_losses['H{}'.format(i)] = [(j & 1) for j in ls]
    bytes_sent['H{}'.format(i)] = bss
    cl = [0]
    for j in ls:
        cl.append(cl[-1]+j)
    cum_losses['H{}'.format(i)] = cl[1:]
    cl = [0]
    for j in bss:
        cl.append(cl[-1]+j)
    cum_bytes_sent['H{}'.format(i)] = cl[1:]
    loss_rates['H{}'.format(i)] = lrs

    
for h in loss_rates:
    variation_loss_rates[h] = []
    variation_loss_rates_xs[h] = []
    lr = loss_rates[h]
    for lri in range(3,len(lr)+1):
        variation_loss_rates[h].append(variation(lr[:lri]))
        variation_loss_rates_xs[h].append(lri)
    

lens = [len(losses[k]) for k in losses]
min_len = min(lens)


#print len(losses['10.0.0.1'][:min_len])
#print len(losses['10.0.0.2'][:min_len])
#print len(losses['10.0.0.3'][:min_len])

pearson_xs = {}
pearson_losses = {}
pearson_loss_rates = {}
cross_corr_losses = {}
cross_corr_loss_rates = {}

hosts = []
for i in range(1,num_hosts+1):
    hosts.append("H{}".format(i))

host_pairs = list(itertools.combinations(hosts,2))
host_pairs_names = ["-".join(p) for p in host_pairs]

print host_pairs_names

for p in host_pairs_names:
    pearson_xs[p] = []
    pearson_losses[p] = []
    pearson_loss_rates[p] = []
    cross_corr_losses[p] = []
    cross_corr_loss_rates[p] = []

if not window_size:
   for i in range(3,min_len+1):
       for p in host_pairs:
           p1 = p[0]
           p2 = p[1]
           pearson_xs[p1+"-"+p2].append(i*1.0)
#           pearson_losses[p1+"-"+p2].append(pearsonr(losses[p1][:i],losses[p2][:i])[0])
           pearson_losses[p1+"-"+p2].append(pearsonr(binarized_losses[p1][:i],binarized_losses[p2][:i])[0])
           pearson_loss_rates[p1+"-"+p2].append(pearsonr(loss_rates[p1][:i],loss_rates[p2][:i])[0])
           cross_corr_losses[p1+"-"+p2].append(crosscorr(binarized_losses[p1][:i],binarized_losses[p2][:i]))
           cross_corr_loss_rates[p1+"-"+p2].append(crosscorr(loss_rates[p1][:i],loss_rates[p2][:i]))
else:
    for i in range(3,min_len+1):
        for p in host_pairs:
            p1 = p[0]
            p2 = p[1]
            # set the 3 threshold to a variable TODO
            if (sum(binarized_losses[p1][max(i-window_size,0):i]) + sum(binarized_losses[p2][max(i-window_size,0):i])) > 3:
                pearson_xs[p1+"-"+p2].append(i*1.0)
                pearson_losses[p1+"-"+p2].append(pearsonr(binarized_losses[p1][max(i-window_size,0):i],binarized_losses[p2][max(i-window_size,0):i])[0])
                pearson_loss_rates[p1+"-"+p2].append(pearsonr(loss_rates[p1][max(i-window_size,0):i],loss_rates[p2][max(i-window_size,0):i])[0])
                cross_corr_losses[p1+"-"+p2].append(crosscorr(binarized_losses[p1][max(i-window_size,0):i],binarized_losses[p2][max(i-window_size,0):i]))
                cross_corr_loss_rates[p1+"-"+p2].append(crosscorr(loss_rates[p1][max(i-window_size,0):i],loss_rates[p2][max(i-window_size,0):i]))
  
x = np.arange(3.0, (min_len+1)*1.0, 1.0)
fig1, ax = plt.subplots()
for d in pearson_losses:
    y = pearson_losses[d]
    ax.plot(pearson_xs[d], y, label=d)
if not window_size:
    ax.set(xlabel='Cumulative number of Windows', ylabel='Pearson correlation coefficient',
            title='Correlation of Losses (Binarized) between Hosts - {}'.format(pcap_name.split("/")[-1]))
else:
    ax.set(xlabel='Cumulative number of Windows', ylabel='Pearson correlation coefficient',
            title='Correlation of Losses (Binarized) between Hosts using past {} windows, {} seconds each - {}'.format(window_size, args.delta, pcap_name.split("/")[-1]))

handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

ax.grid()
ax.legend(handles, labels)


fig2, ax = plt.subplots()
for d in cum_losses:
    y = cum_losses[d][3:min_len]
    ax.plot(x[:-1], y, label=d)
ax.set(xlabel='Window number', ylabel='Cumulative # of Losses',
        title='Cumulative Number of Losses at each Window - {}'.format(pcap_name.split("/")[-1]))
    
handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

ax.grid()
ax.legend()

fig3, ax = plt.subplots()
for d in losses:
    y = losses[d][3:min_len]
    ax.plot(x[:-1], y, label=d)
ax.set(xlabel='Window number', ylabel='# of Losses',
        title='Number of Losses at each Window - {}'.format(pcap_name.split("/")[-1]))
    
handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

ax.grid()
ax.legend()


fig4, ax = plt.subplots()
for d in cum_bytes_sent:
    y = cum_bytes_sent[d][3:min_len]
    ax.plot(x[:-1], y, label=d)
ax.set(xlabel='Window number', ylabel='Cumulative Bytes Sent per Window',
        title='Cumulative Bytes in each Window - {}'.format(pcap_name.split("/")[-1]))
    
handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

ax.grid()
ax.legend()

fig5, ax = plt.subplots()
for d in bytes_sent:
    y = bytes_sent[d][3:min_len]
    ax.plot(x[:-1], y, label=d)
ax.set(xlabel='Window number', ylabel='Bytes Sent per Window',
        title='Bytes Sent in each Window - {}'.format(pcap_name.split("/")[-1]))
    
handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

ax.grid()
ax.legend()

x = np.arange(3.0, (min_len+1)*1.0, 1.0)
fig6, ax = plt.subplots()
for d in pearson_loss_rates:
    y = pearson_loss_rates[d]
    ax.plot(pearson_xs[d], y, label=d)
if not window_size:
    ax.set(xlabel='Cumulative number of Windows', ylabel='Pearson correlation coefficient',
            title='Correlation of Losses Rates between Hosts - {}'.format(pcap_name.split("/")[-1]))
else:
    ax.set(xlabel='Cumulative number of Windows', ylabel='Pearson correlation coefficient',
            title='Correlation of Loss Rates between Hosts using past {} windows, {} seconds each - {}'.format(window_size, args.delta, pcap_name.split("/")[-1]))

handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

ax.grid()
ax.legend(handles, labels)

fig7, ax = plt.subplots()
for d in variation_loss_rates:
    x = variation_loss_rates_xs[d]
    y = variation_loss_rates[d]
    ax.plot(x, y, label=d)
#if not window_size:
ax.set(xlabel='Cumulative number of Windows', ylabel='coefficient of variation',
       title='Variation of Losses Rates between Hosts - {}'.format(pcap_name.split("/")[-1]))
#else:
#    ax.set(xlabel='Cumulative number of Windows', ylabel='coefficient of variation',
#            title='Variation of Loss Rates between Hosts using past {} windows, {} seconds each - {}'.format(window_size, args.delta, pcap_name.split("/")[-1]))

handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

ax.grid()
ax.legend(handles, labels)

x = np.arange(3.0, (min_len+1)*1.0, 1.0)
fig8, ax = plt.subplots()
for d in cross_corr_losses:
    y = cross_corr_losses[d]
    ax.plot(pearson_xs[d], y, label=d)
if not window_size:
    ax.set(xlabel='Cumulative number of Windows', ylabel='cross correlation coefficient',
            title='Cross Correlation of Losses (Binarized) between Hosts - {}'.format(pcap_name.split("/")[-1]))
else:
    ax.set(xlabel='Cumulative number of Windows', ylabel='cross correlation coefficient',
            title='Cross Correlation of Loss (Binarized) between Hosts using past {} windows, {} seconds each - {}'.format(window_size, args.delta, pcap_name.split("/")[-1]))

handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

ax.grid()
ax.legend(handles, labels)

x = np.arange(3.0, (min_len+1)*1.0, 1.0)
fig9, ax = plt.subplots()
for d in cross_corr_loss_rates:
    y = cross_corr_loss_rates[d]
    ax.plot(pearson_xs[d], y, label=d)
if not window_size:
    ax.set(xlabel='Cumulative number of Windows', ylabel='cross correlation coefficient',
            title='Cross Correlation of Losses Rates between Hosts - {}'.format(pcap_name.split("/")[-1]))
else:
    ax.set(xlabel='Cumulative number of Windows', ylabel='Pearson correlation coefficient',
            title='Cross Correlation of Loss Rates between Hosts using past {} windows, {} seconds each - {}'.format(window_size, args.delta, pcap_name.split("/")[-1]))

handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

ax.grid()
ax.legend(handles, labels)

fig10, ax = plt.subplots()
for d in binarized_losses:
    y = binarized_losses[d][3:min_len]
    ax.plot(x[:-1], y, label=d)
ax.set(xlabel='Window number', ylabel='# of Losses (Binarized)',
        title='Losses at each Window (Binarized) - {}'.format(pcap_name.split("/")[-1]))
    
handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

ax.grid()
ax.legend()

fig11, ax = plt.subplots()
for d in loss_rates:
    y = loss_rates[d][3:min_len]
    ax.plot(x[:-1], y, label=d)
ax.set(xlabel='Window number', ylabel='Loss Rate',
        title='Loss Rate for Each Window - {}'.format(pcap_name.split("/")[-1]))
    
handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

ax.grid()
ax.legend()



#fig1.savefig(path + "/pearson-losses.png")
fig1.savefig(path + "/pearson-losses-binarized.png")
fig2.savefig(path + "/cum-losses.png")
fig3.savefig(path + "/losses.png")
fig4.savefig(path + "/cum-bytes.png")
fig5.savefig(path + "/bytes.png")
fig6.savefig(path + "/pearson-loss-rates.png")
fig7.savefig(path + "/variation-loss-rates.png")
fig8.savefig(path + "/cross-corr-losses.png")
fig9.savefig(path + "/cross-corr-loss-rates.png")
fig10.savefig(path + "/binarized-losses.png")
fig11.savefig(path + "/loss-rates.png")

#fig1.savefig("./results/graphs/{}-pearson-losses.png".format((pcap_name.split("/")[-1]).split('.pcapng')[-2]))
fig1.savefig("./results/graphs/{}-pearson-losses-binarized.png".format((pcap_name.split("/")[-1]).split('.pcapng')[-2]))
fig2.savefig("./results/graphs/{}-cum-losses.png".format((pcap_name.split("/")[-1]).split('.pcapng')[-2]))
fig3.savefig("./results/graphs/{}-losses.png".format((pcap_name.split("/")[-1]).split('.pcapng')[-2]))
fig4.savefig("./results/graphs/{}-cum-bytes.png".format((pcap_name.split("/")[-1]).split('.pcapng')[-2]))
fig5.savefig("./results/graphs/{}-bytes.png".format((pcap_name.split("/")[-1]).split('.pcapng')[-2]))
fig6.savefig("./results/graphs/{}-pearson-loss-rates.png".format((pcap_name.split("/")[-1]).split('.pcapng')[-2]))
fig7.savefig("./results/graphs/{}-variation-loss-rates.png".format((pcap_name.split("/")[-1]).split('.pcapng')[-2]))
fig8.savefig("./results/graphs/{}-cross-corr-losses.png".format((pcap_name.split("/")[-1]).split('.pcapng')[-2]))
fig9.savefig("./results/graphs/{}-cross-corr-loss-rates.png".format((pcap_name.split("/")[-1]).split('.pcapng')[-2]))
fig10.savefig("./results/graphs/{}-binarized-losses.png".format((pcap_name.split("/")[-1]).split('.pcapng')[-2]))
fig11.savefig("./results/graphs/{}-loss-rates.png".format((pcap_name.split("/")[-1]).split('.pcapng')[-2]))

# fig3, ax3 = plt.subplots()
# for d in pearson_loss_rates:
#     y = pearson_loss_rates[d]
#     ax3.plot(x, y, label=d)
# ax3.set(xlabel='Window number', ylabel='Pearson correlation coefficient',
#        title='Correlation of Loss Rates between Hosts - {}'.format(pcap_name.split("/")[-1]))
# ax3.grid()
# ax3.legend()
# 
# fig4, ax4 = plt.subplots()
# for d in loss_rates:
#     y = loss_rates[d][3:min_len]
#     ax4.plot(x[:-1], y, label=d)
# ax4.set(xlabel='Window number', ylabel='Loss Rates',
#         title='Loss Rates during each Window - {}'.format(pcap_name.split("/")[-1]))
# ax4.grid()
# ax4.legend()
# 
# fig3.savefig(path + "/pearson-loss-rates.png")
# fig4.savefig(path + "/loss-rates.png")
# 
# fig3.savefig("./results/graphs/{}-pearson-loss-rates.png".format((pcap_name.split("/")[-1]).split('.pcapng')[-2]))
# fig4.savefig("./results/graphs/{}-loss-rates.png".format((pcap_name.split("/")[-1]).split('.pcapng')[-2]))
# 
# 
plt.show(fig10)
