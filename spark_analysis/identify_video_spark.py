import pprint
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from fields import fields
import sys


pp = pprint.PrettyPrinter(indent=2)

sc = SparkContext("local", "Gilberto PCAP")

RDDread = sc.textFile(sys.argv[1]).map(lambda x: x.split('|'))

# remove the '.' and '_'  and '-' in the field names
fs = [f.replace('.','').replace('_','').replace('-','') for f in fields]

RowBuilder = Row(*fs)
table = RDDread.map(lambda p: RowBuilder(*p))

spark = SparkSession.builder.appName('Analysis').getOrCreate()

tabledf = spark.createDataFrame(table)

#output = RDDread.first()

def col(s):
    if s in fs:
        print "Returning '{}'".format(fs.index(s))
        return fs.index(s)
    else:
        "Unknown Column Name: {}".format(s)
        sys.exit()

print '-'*79
print '-'*79

# Query to print out DNS queries/answers to netflix and youtube
tabledf.where((tabledf.dnsqryname.like("%netflix%")) | (tabledf.dnsqryname.like("%youtube%"))).show(20,truncate=False)

# Print a TCP Session greater than a certain frame number for 2 IPs
#tabledf.where((tabledf.ipsrchost == '191.185.42.118') & (tabledf.ipdsthost == '201.17.128.197')).show(20,truncate=False)

print '-'*79
print '-'*79
