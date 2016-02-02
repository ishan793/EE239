import pickle
import csv

from sklearn.preprocessing import OneHotEncoder

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
	
	for i in d:
		try:
			res.append(int(i))
		except ValueError:
			res.append(float(i))
	
	return res

def get_num(x):
	res = []
	for i in x:
		res.append(get_vec(i))
	return res


# function to convert a csv file to pickle
def convert_network(filename,final_filename, var_flag = 0):
	'''
	Filename : input filename of csv filename
	final_filename : o/p filename of .pickle file
	'''
	res = {'x':[],'y':[]}
	with open(filename,'rb') as csvfile:
		f = csv.reader(csvfile)
		count = 0
		for line in f:
			if count != 0:
				if var_flag == 0:
					res['x'].append(line[:-2]+[line[-1]])
					res['y'].append(float(line[-2]))
				else:
					res['x'].append(line[:-1])
					res['y'].append(float(line[-1]))
			count += 1

	res['x'] = get_num(res['x'])
	m = len(res['x'][0])-1
	enc = OneHotEncoder(categorical_features = range(m),sparse = False)
	enc.fit(res['x'])
	res['x'] = enc.transform(res['x'])
	
	with open(final_filename,'wb') as f:
		pickle.dump(res,f)

convert_network('network_backup_dataset.csv','network_data_size.pickle')
convert_network('network_backup_dataset.csv','network_backup_time.pickle', var_flag = 1)
convert_pickle('housing_data.csv','housing_data.pickle')
