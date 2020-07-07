import csv
import sys
import math

filename = sys.argv[1]

rows = []                                         
with open(filename, 'r') as csvfile:          
    reader = csv.reader(csvfile, delimiter='|')
    for row in reader:        
        rows.append(row)

end_time = float(rows[-1][0])

# Set the time interval Here
#interval = int(math.ceil(end_time / 4.0))
interval = 1

num_intervals = int(math.ceil(end_time / interval))

def b2mbits(v):
    return (v*8*1.0)/1000000

datapoints = []
curr_row_num = 0
cur_start_time = float(rows[0][0])
cur_end_time = float(rows[0][0])
for curr_interval_num in range(num_intervals):
    if curr_row_num >= len(rows):
        break
    row = rows[curr_row_num]
    cum_sum = 0    
    while ((curr_interval_num + 1) * (interval*1.0)) > float(row[0]):
        if curr_row_num >= len(rows):
            break
        row = rows[curr_row_num]
        cum_sum += b2mbits(int(row[1]))
        curr_row_num += 1
#    cur_end_time = float(row[0])

    cur_throughput = (cum_sum*1.0) / interval
    datapoints.append((curr_interval_num*interval, cur_throughput))

#    print "{}\tSum: {}, Time: {}, Tput: {}".format(curr_interval_num, cum_sum, cur_end_time - cur_start_time, cur_throughput)
        

for d in datapoints:
    print str(d[0]) + '\t' + str(d[1])

        
total_size = 0
for row in rows:
    total_size += b2mbits(int(row[1]))

total_avg_throughput = (total_size * 1.0) / end_time

print "Total Avg Throughput: {} Mbits/sec".format(total_avg_throughput)

