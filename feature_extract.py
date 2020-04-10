import pandas as pd
import numpy as np
import math

data_feature_ALL = []

def load_data(x, y):
	while x < y:
		c_path = './B0006/B0006_' + str(x) + '_charge_24.csv'
		dc_path = './B0006/B0006_' + str(x + 1) + '_discharge_24.csv'
		pre_data(pd.read_csv(c_path), pd.read_csv(dc_path))
		x += 2
	print('read finish...to' + str(y))

def load_data_2(x, y):
	while x < y:
		c_path = './B0006/B0006_' + str(x) + '_charge_24.csv'
		dc_path = './B0006/B0006_' + str(x + 2) + '_discharge_24.csv'
		pre_data(pd.read_csv(c_path), pd.read_csv(dc_path))
		x += 4
	print('read finish...to' + str(y))

def pre_data(c_data, dc_data):
	data_feature = []
	c_data = c_data.to_numpy()
	dc_data = dc_data.to_numpy()
	# charge process
	# (1)
	for f in c_data:
		if f[0] > 4.2:
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
	pos = len(dc_data) - 1
	data_feature.append(dc_data[pos][5])
	data_feature.append(dc_data[pos][0])
	# (7)
	for i in range(2, len(dc_data)):
		if dc_data[i][1] > -0.1:
			data_feature.append(dc_data[i][5])
			data_feature.append(dc_data[i][1])
			break
		elif i == (len(dc_data) - 1):
			data_feature.append(dc_data[i][5])
			data_feature.append(0.0)
			break
	# (8)
	vm_max_2 = np.max(dc_data, axis=0)[2]
	for f in dc_data:
		if f[2] == vm_max_2:
			data_feature.append(f[5])
			data_feature.append(f[2])
	# (9)
	for i in range(2, len(dc_data)):
		if i == len(dc_data) - 1:
			data_feature.append(dc_data[i][5])
			data_feature.append(0.0)
		elif dc_data[i][3] < 0:
			if dc_data[i][3] > -1:
				data_feature.append(dc_data[i][5])
				data_feature.append(dc_data[i][3])
				break
		elif dc_data[i][3] > 0:
			if dc_data[i][3] < 1:
				data_feature.append(dc_data[i][5])
				data_feature.append(0.0 - dc_data[i][3])
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
	t = dc_data[len(dc_data) - 1][5] - dc_data[2][5]
	Q = t/60 * 2.0
	data_feature.append(Q)
	# 308 = sample size
	# RUL = 308 + 1 -(math.ceil(cycle / 2))
	# data_feature.append(RUL)
	data_feature_ALL.append(data_feature)

if __name__ == '__main__':
	load_data(1, 22)
	load_data(24, 39)
	load_data_2(40, 82)
	load_data(85, 86)
	load_data_2(88, 134)
	load_data(135, 136)
	load_data_2(138, 148)
	load_data(149, 150)
	load_data_2(152, 214)
	load_data(215, 216)
	load_data_2(218, 264)
	load_data(265, 266)
	load_data_2(268, 310)
	load_data(315, 316)
	load_data_2(318, 364)
	load_data(366, 367)
	load_data_2(369, 431)
	load_data(432, 433)
	load_data_2(435, 481)
	load_data(482, 485)
	load_data_2(487, 545)
	load_data(547, 548)
	load_data_2(550, 612)
	load_data(613, 614)
	# print(len(data_feature_ALL))

	# 添加对应RUL
	RUL = []
	for i in range(167):
		RUL.append((167 - i) / 167)

	# 添加对应soh
	soh_list = [1.04]
	soh_data = pd.read_csv('B0006.csv').to_numpy()
	for h in range(1,166):
		soh_list.append(soh_data[h][8])
	soh_list.append(0.59)

	# print(len(data_feature_ALL))
	# print(len(soh_list))

	for i in range(167):
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
		t += 1
	# print(max_f)
	# print(min_f)

	for f in data_feature_ALL:
		for i in range(21):
			f[i] = (f[i] - min_f[i]) / (max_f[i] - min_f[i])
	# print(data_feature_ALL)
	print(len(data_feature_ALL))

	# 写入.csv文件

	_csv = pd.DataFrame(data = data_feature_ALL)
	csv_path = 'D:/pred_nasa/B0006_feature.csv'
	_csv.to_csv(csv_path, index = False, header = False)
