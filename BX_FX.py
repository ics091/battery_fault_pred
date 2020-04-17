import pandas as pd
import numpy as np
import math

data_feature_ALL = []

def load_data(x):
	i = 0
	while i < x + 1:
		c_path = './B0007/B0007_charge' + str(i) + '.csv'
		dc_path = './B0007/B0007_discharge' + str(i) + '.csv'
		pre_data(pd.read_csv(c_path), pd.read_csv(dc_path))
		i += 1
	print('read_finish')

def pre_data(c_data, dc_data):
	data_feature = []
	c_data = c_data.to_numpy()
	dc_data = dc_data.to_numpy()
	# charge process
	# (1)
	for f in c_data:
		if f[0] > 4.19999:
			data_feature.append(f[5])
			data_feature.append(f[0])
			break
	# (2)
	for i in range(2, len(c_data)):
		if c_data[i][1] < 1.50:
			data_feature.append(c_data[i][5])
			data_feature.append(c_data[i][1])
			break
	# (3)
	tmp_max = np.max(c_data, axis=0)[2]
	for f in c_data:
		if f[2] == tmp_max:
			data_feature.append(f[5])
			data_feature.append(f[2])
			break
	# (4)
	for i in range(2, len(c_data)):
		if c_data[i][3] < 1.50:
			data_feature.append(c_data[i][5])
			data_feature.append(c_data[i][3])
			break
	# (5)
	vm_max_1 = np.max(c_data, axis=0)[4]
	for f in c_data:
		if f[4] == vm_max_1:
			data_feature.append(f[5])
			data_feature.append(f[4])
			break
	# discharge process
	# (6)
	vm_min_1 = np.min(dc_data, axis=0)[0]
	for f in dc_data:
		if f[0] == vm_min_1:
			data_feature.append(f[5])
			data_feature.append(f[0])
			break
	# (7)
	for i in range(2, len(dc_data)):
		if dc_data[i][1] > -1.0:
			data_feature.append(dc_data[i][5])
			data_feature.append(dc_data[i][1])
			break
		elif i == (len(dc_data) - 1):
			data_feature.append(dc_data[i][5])
			data_feature.append(0.0)
			break
	# (8)
	tmp_max_2 = np.max(dc_data, axis=0)[2]
	for f in dc_data:
		if f[2] == tmp_max_2:
			data_feature.append(f[5])
			data_feature.append(f[2])
			break
	# (9)
	if dc_data[2][3] > 0:
		for i in range(2,len(dc_data)):
			if dc_data[i][3] < 0.1:
				data_feature.append(dc_data[i][5])
				data_feature.append(0.0 - dc_data[i][3])
				break
			elif i == (len(dc_data) - 1):
				data_feature.append(dc_data[i][5])
				data_feature.append(0.0)
				break
	elif dc_data[2][3] < 0:
		for i in range(2,len(dc_data)):
			if dc_data[i][3] > -0.1:
				data_feature.append(dc_data[i][5])
				data_feature.append(dc_data[i][3])
				break
			elif i == (len(dc_data) - 1):
				data_feature.append(dc_data[i][5])
				data_feature.append(0.0)
				break
	# (10)
	k = len(dc_data) - 1
	while k > 0:
		if dc_data[k][4] > 0:
			data_feature.append(dc_data[k][5])
			data_feature.append(dc_data[k][4])
			break
		else:
			pass
		k = k - 1
	# (11)
	data_feature.append(dc_data[0][6])
	data_feature_ALL.append(data_feature)

if __name__ == '__main__':
	load_data(164)

	# for x in data_feature_ALL:
	# 	print(x[16])
	# 	print(x[17])
	# 	print('---')
	# 添加对应RUL
	RUL = []
	for i in range(0,165):
		RUL.append((165 - i) / 165)

	# 添加对应soh
	soh_list = []
	soh_data = pd.read_csv('B0007.csv').to_numpy()
	for h in range(165):
		soh_list.append(soh_data[h][8])

	for i in range(165):
		data_feature_ALL[i].append(RUL[i])
		data_feature_ALL[i].append(soh_list[i])



	# 将特征放缩到 （0， 1）的范围

	data_feature_ALL = np.array(data_feature_ALL)
	max_f = []
	min_f = []

	t = 0

	while t < 21:
		max_f.append(np.max(data_feature_ALL, axis=0)[t])
		min_f.append(np.min(data_feature_ALL, axis=0)[t])
		print('no' + str(t) + '=', max_f[t] - min_f[t])
		t += 1
	for f in data_feature_ALL:
		for i in range(21):
			f[i] = (f[i] - min_f[i]) / (max_f[i] - min_f[i])

	#写入.csv文件

	_csv = pd.DataFrame(data = data_feature_ALL)
	csv_path = 'D:/pred_nasa/B0007_feature.csv'
	_csv.to_csv(csv_path, index = False, header = False)