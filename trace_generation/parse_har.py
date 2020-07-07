import json
import sys
import pprint
from datetime import datetime

def Record(request_time, response_time, request_size, range_start, range_end, server_ip, src_port, url):
    d = {}
    d['request_time'] = request_time
    d['response_time'] = response_time
    d['request_size'] = request_size
    d['range_start'] = range_start
    d['range_end'] = range_end
    d['unknown_bitrate'] = None
    d['server_ip'] = server_ip
    d['src_port'] = src_port
    d['url'] = url
    return d

har_file = sys.argv[1]
if '/' in sys.argv[1]:
    directory = sys.argv[1].split('/')[:-1]
else:
    directory = './'

# get the timestamp file
# TODO: replace i_utc_time
    
d = json.loads(open(har_file).read())

records = []

for i in range(len(d['log']['entries'])):
    url = d['log']['entries'][i]['request']['url']
    if "/range/" in url:
#        pprint.pprint(d['log']['entries'][i])
        for cd in d['log']['entries'][i]['request']['queryString']:
            if cd['name'] == "range":
                print cd['value']
                break
#        pprint.pprint(d['log']['entries'][i])
        time = d['log']['entries'][i]['startedDateTime']

        
        size = d['log']['entries'][i]['response']['bodySize'] + d['log']['entries'][i]['response']['headersSize']
        
        utc_time = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ")
        epoch_time = (utc_time - datetime(1970, 1, 1)).total_seconds()

        end_time = epoch_time + (d['log']['entries'][i]['time'] / 1000.0)

        # replace with init time for each
        
        i_utc_time = datetime.strptime("2020-04-17T02:23:30.345Z", "%Y-%m-%dT%H:%M:%S.%fZ")
        i_epoch_time = (i_utc_time - datetime(1970, 1, 1)).total_seconds()

        # replace with init time from timestamp file
        i_epoch_time = 1587090210.12
        
        ranges = url.split('/range/')[1].split('?')[0].split('-')
        
        range_start = int(ranges[0])
        range_end = int(ranges[1])

        src_port = None
        
        headers = d['log']['entries'][i]['response']['headers']
        for h in headers:
            if h['name'] == 'X-TCP-Info':
                src_port = int(h['value'].split('port=')[1])

        server_ip = d['log']['entries'][i]['serverIPAddress']
        
        if (epoch_time-i_epoch_time) > 0.0:
            records.append(Record(epoch_time-i_epoch_time, end_time-i_epoch_time, size, range_start, range_end, server_ip, src_port, url))


# label the data with unknown bitrates based on the range values
clabels='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890/+-'
brates = []
for r in records:
    range_start = r['range_start']
    range_end = r['range_end']

    added = False
    for i in range(len(brates)):
        br = brates[i]
        c_end = br[-1][1]
        if range_start == (c_end+1):
            br.append((range_start,range_end))
            r['unknown_bitrate'] = clabels[i]
            added = True
            break

    if not(added):
        r['unknown_bitrate'] = clabels[len(brates)]
        brates.append([(range_start, range_end)])




        
for r in records:
#    print str(r['request_time']) + "\t" + str(r['response_time']) + "\t" + str(r['request_size']) + "\t" + str(r['range_start']) + "\t" + str(r['range_end']) + "\t" + r['unknown_bitrate'] + "\t" + r['server_ip'] + "\t" + str(r['src_port']) + "\t" + str(r['url'])
    print str(r['request_time']) + "\t" + str(r['response_time']) + "\t" + str(r['request_size']) + "\t" + r['server_ip'] + "\t" + str(r['src_port'])

    
        
#        print "-"*79



        
