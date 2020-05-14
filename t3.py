import pandas as pd
import numpy as np
import torch

def path(time):
	h_data_path = './' + time + '/h/h'
	nh_data_path = './' + time + '/nh/nh'
	return h_data_path, nh_data_path

def load_HNH(l1, l2, time):
	p1, p2 = path(time)
	x = 1
	y = 1
	cb_data = []
	label = []
	while x <= l1:
		try:
			p = p1 + str(x) + '.csv'
			data = pd.read_csv(p, header=None)
			data = data.iloc[:, 2:18].values.tolist()
			cb_data.append(data)
			label.append(1)
		except:
			pass
		x += 1

	while y <= l2:
		try:
			p = p2 + str(y) + '.csv'
			data = pd.read_csv(p, header=None)
			data = data.iloc[:, 2:18].values.tolist()
			cb_data.append(data)
			label.append(0)
		except:
			pass
		y += 1
	return cb_data, label

def load_data():
	time_list = ['2018-11','2018-12','2019-01','2019-08','2019-09','2019-10']
	data_1811, label_1811 = load_HNH(52,117,time_list[0])
	data_1812, label_1812 = load_HNH(40,69,time_list[1])
	data_1901, label_1901 = load_HNH(10,32,time_list[2])

	X1 = []
	y1 = []

	for x, y in zip(data_1811, label_1811):
		X1.append(x)
		y1.append(y)

	for x, y in zip(data_1812, label_1812):
		X1.append(x)
		y1.append(y)

	for x, y in zip(data_1901, label_1901):
		X1.append(x)
		y1.append(y)

	for x in X1:
		x = torch.from_numpy(np.array(x))

	print(len(X1))


if __name__ == '__main__':
	load_data()