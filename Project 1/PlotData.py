# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 20:07:52 2016

@author: shubham
"""

import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
import itertools as itertools

# function to convert a csv file to pickle
def convert_pickle(filename, final_filename):
	'''
	Filename : input filename of csv filename
	final_filename : o/p filename of .pickle file
	'''
	res = {'x':[],'y':[]}
	with open(filename,'rb') as csvfile:
		f = csv.reader(csvfile)
		for line in f:
			res['x'].append(line[:-1])
			res['y'].append(line[-1])
	
	with open(final_filename,'wb') as f:
		pickle.dump(res,f)


def get_vec(d):
	weeks = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
	res = []
	d[1] = weeks[d[1]]
	d[3] = (d[3][10:])
	d[4] = (d[4][5:])
	res = [int(i) for i in d]
	return res

def get_num(x):
	res = []
	for i in x:
		res.append(get_vec(i))
	return res 
 
def find_hour(x):
	return ((x[0])*24)+x[2]

def find_amount(x):
		return x[6]
	
 
 
#------------------------------------------------------------------------------
# read the csv file and scale the label for float conversion
res = {'x':[],'y':[]}
with open('network_backup_dataset.csv','rb') as csvfile:
		f = csv.reader(csvfile)
		count = 0
		for line in f:
			if count != 0:
				res['x'].append(line[:-2]+[line[-1]]+[np.asarray(line[-2]).astype(np.float)*10000])
			count += 1

	
	
res['x'] = get_num(res['x'])

file_range = 30
final_res = {}
for i in range(5):
    final_res[i] = {}
    for j in range(file_range):
        final_res[i][j] = {}
        for k in range(5):
            final_res[i][j][k]= {'x':[],'y':[]}


#------------------------------------------------------------------------------
#divide data based on workflow ID

data =np.asarray(res['x']).astype(np.float)
data[:,0]=(data[:,0]-1)*7+data[:,1]
data=data[np.argsort(data[:,3])]
data0=[]
data0.append(data[data[:,3]==0])
data0.append(data[data[:,3]==1])
data0.append(data[data[:,3]==2])
data0.append(data[data[:,3]==3])
data0.append(data[data[:,3]==4])
data = []
#------------------------------------------------------------------------------
for i in range(5):
    # i is the workflow_id
    data = data0[i]
    for trans_array in data:    
        file_id = trans_array[4]
        trans_hour = find_hour(trans_array)
        data_trans = find_amount(trans_array)
        for k in range(5):
            if((trans_hour>(24*20*k))& (trans_hour<(24*20*(k+1)))):
                final_res[i][file_id][k]['x'].append(trans_hour)
                final_res[i][file_id][k]['y'].append(data_trans)

colors = itertools.cycle(["r", "b", "g"])
for i in range(1):
        for k in range(5):
            plt.figure(k)
            for j in range(30):
                plt.scatter(final_res[i][j][k]['x'],final_res[i][j][k]['y'],c=np.random.rand(3,1));    