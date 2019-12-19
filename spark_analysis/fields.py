#!/usr/bin/python

fields = ['frame.number',
          'frame.time',
          'ip.proto',
          'ip.src_host',
          'tcp.srcport',
          'udp.srcport',
          'ip.dst_host',
          'tcp.dstport',
          'udp.dstport',
          'tcp.len',
          'udp.length',
          'dns.qry.name',
          'dns.flags.response',
          'dns.a'
          '']

fields = filter(lambda a: a != '', fields)
