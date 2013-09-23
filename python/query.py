# query <master> <inputFile> <inds> <outputFile>
# 
# quickly query a data set by averaging together points
# each row is (x,y,z,timeseries)
#

import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 4:
  print >> sys.stderr, \
  "(query) usage: query <master> <inputFile> <inds> <outputFile>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	k = int(vec[0]) + int((vec[1] - 1)*1235) + int((vec[2] - 1)*1248*1235)
	med = median(ts)
	ts = (ts - med) / (med + 0.1) # convert to dff
	return (k,ts)

# parse inputs
sc = SparkContext(sys.argv[1], "query")
inputFile = str(sys.argv[2])
indsFile = str(sys.argv[3])
outputFile = str(sys.argv[4]) + "-query"
logging.basicConfig(filename=outputFile+'-stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

logging.info("(query) loading data")
data = sc.textFile(inputFile).map(parseVector) # the data

inds = loadmat(indsFile)['inds']
n = len(inds)

ts = data.filter(lambda (k,x) : k in inds).map(lambda (k,x) : x).reduce(lambda x,y :x+y) / n

savemat(outputFile+"-ts.mat",mdict={'ts':ts},oned_as='column',do_compression='true')


