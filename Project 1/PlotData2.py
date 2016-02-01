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
	res = {'x':[],'y':[],'yp':[]}
	with open(filename,'rb') as csvfile:
		f = csv.reader(csvfile)
		for line in f:
			res['x'].append(line[:-1])
			res['y'].append(line[-2])
                res['yp'].append(line[-1])
	
	with open(final_filename,'wb') as f:
		pickle.dump(res,f)


def get_vec(d):
	weeks = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
	res = []
	d[1] = weeks[d[1]]
	d[3] = (d[3][10:])
	d[4] = (d[4][5:])
	res = [float(i) for i in d]
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
res = {'x':[],'y':[],'yp':[]}
with open('network_backup_dataset.csv','rb') as csvfile:
		f = csv.reader(csvfile)
		count = 0
		for line in f:
			if count != 0:
				res['x'].append(line[:-3]+[line[-2]]+[np.asarray(line[-3]).astype(np.float)]+[line[-1]])
			count += 1

	
	
res['x'] = get_num(res['x'])

file_range = 30
final_res = {}
out = {}
for i in range(5):
    # i is for work flow id
    final_res[i] = {}
    out[i]={}
    for k in range(6):
        # k indicates the day period
        final_res[i][k]= {}
        out[i][k]={}

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
val = 24*20

for i in range(5):
    # i is the workflow_id
    data = data0[i]
    for trans_array in data:    
        # file_id = trans_array[4]
        trans_hour = find_hour(trans_array)
        data_trans = find_amount(trans_array)
        data_out = trans_array[7]

        k = int((trans_hour-1)/val)
        ind = (trans_hour-1)%val
        if ind not in final_res[i][k]:
            final_res[i][k][ind] = 0
            out[i][k][ind]=0
        final_res[i][k][ind] += data_trans
        out[i][k][ind] +=data_out
        '''for k in range(5):
            if((trans_hour>(24*20*k))& (trans_hour<(24*20*(k+1)))):
                if trans_hour not in final_res[i][k]:
                    final_res[i][k][trans_hour] = 0
                final_res[i][k][trans_hour] += data_trans
        '''
plot_res = {}
for i in range(5):
    # i is for work flow id
    plot_res[i] = {}
    for k in range(6):
        # k indicates the day period
        plot_res[i][k]= {'x':[],'y':[],'yp':[]}
for i in final_res:
    # i is the wf_id    
    for k in final_res[i]:    
    # k is the session number
        for time_stamp in final_res[i][k]:
            plot_res[i][k]['x'].append(time_stamp)
            plot_res[i][k]['y'].append(final_res[i][k][time_stamp])
            plot_res[i][k]['yp'].append(out[i][k][time_stamp])
            
c = ["r", "b", "g"]
f_name = '' 
for i in range(5):
        for k in range(5):
            plt.figure()
            plt.scatter(plot_res[i][k]['x'],(np.asarray(plot_res[i][k]['y']).astype(np.float)),c="r") 
            #plt.scatter(plot_res[i][k]['x'],(np.asarray(plot_res[i][k]['y']).astype(np.float)-np.asarray(plot_res[i][k]['yp']).astype(np.float)),c="r")    
            plt.scatter(plot_res[i][k]['x'],plot_res[i][k]['yp'],c="b",alpha=0.5)
            plt.xlabel('Time(Hr)')
            plt.ylabel('Copy Size')
            #plt.legend('Residue')
            plt.legend('Random Forest: Copy Size vs Time')            
            st = 'resr_f_wf_'+str(i)+'_k_'+str(k)+'.png'
            plt.savefig(st)