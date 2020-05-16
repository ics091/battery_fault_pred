import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import random

SEED = 1024
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

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
			data = data.iloc[:, 1:15].values.tolist()
			cb_data.append(data)
			label.append(1)
		except:
			pass
		x += 1

	while y <= l2:
		try:
			p = p2 + str(y) + '.csv'
			data = pd.read_csv(p, header=None)
			data = data.iloc[:, 1:15].values.tolist()
			cb_data.append(data)
			label.append(0)
		except:
			pass
		y += 1
	return cb_data, label

def normalize(X):
	max_V = []
	min_V = []
	for x in X:
		max_ = []
		min_ = []
		ft_length = np.array(x).shape[1]
		for i in range(ft_length):
			max_.append(np.max(np.array(x), axis=0)[i])
			min_.append(np.min(np.array(x), axis=0)[i])
		max_V.append(max_)
		min_V.append(min_)

	max_V = np.array(max_V)
	min_V = np.array(min_V)
	# print(np.array(max_V).shape,np.array(min_V).shape)
	max_l = []
	min_l = []
	for i in range(max_V.shape[1]):
		max_l.append(np.max(max_V, axis=0)[i])
		min_l.append(np.min(min_V, axis=0)[i])

	for x in X:
		for t in x:
			for i in range(len(t)):
				t[i] = (t[i] - min_l[i]) / (max_l[i] - min_l[i])

	return X

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

	X1 = normalize(X1)

	X1_tensor = []

	for x in X1:
		x = torch.Tensor(x).float()
		X1_tensor.append(x)

	# y1 = torch.from_numpy(np.array(y1)).float()

	return X1_tensor, y1

class DataSET(Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y

	def __len__(self):
		return len(self.X)

	def __getitem__(self, index):
		return self.X[index], self.y[index]

def collate_fn(dataset):
	dataset.sort(key=lambda data: len(data[0]), reverse=True)
	X = []
	y = []
	for data in dataset:
		X.append(data[0])
		y.append(data[1])

	data_length = [len(data) for data in X]
	X = rnn_utils.pad_sequence(X, batch_first=True, padding_value=0)
	y = torch.from_numpy(np.asarray(y))
	return X, y, data_length

# pred result to one zero
def to_one_zero(X):
	for i in range(len(X)):
		if X[i][0] >= 0.5:
			X[i][0] = 1
		else:
			X[i][0] = 0
	X = X.reshape(-1,)
	return X

# LSTM model
class LSTM(nn.Module):
	def __init__(self):
		super(LSTM, self).__init__()
		self.rnn = nn.LSTM(
			input_size=INTPUT_SIZE,
			hidden_size=24,
			num_layers=2,
			batch_first=True,
			)
		self.dense= nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(24, 12),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(12, 6),
			nn.ReLU(),
			nn.Linear(6, 1)
			)

	def forward(self, x):
		# out = self.linear_layers(x)
		h0 = c0 = torch.randn(2, x[1].tolist()[0], 24)
		r_out, h_n = self.rnn(x, (h0, c0))
		# r_out, out_len = rnn_utils.pad_packed_sequence(r_out, batch_first=True)
		out = h_n[0][1]
		out = self.dense(out)
		return out

BATCH_SIZE = 3
INTPUT_SIZE = 14
LR = 0.01

if __name__ == '__main__':
	# split trian test
	X1, y1 = load_data()
	X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.1)
	y1_train = torch.from_numpy(np.array(y1_train)).float()
	y1_test = torch.from_numpy(np.array(y1_test)).float()

	# train_data loader
	dataSet_1 = DataSET(X1_train,y1_train)
	train_loader_1 = Data.DataLoader(
		dataset=dataSet_1,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=2,
		collate_fn = collate_fn
		)

	# model
	lstm = LSTM()
	optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
	loss_func = nn.BCEWithLogitsLoss()

	for epoch in range(30):
		for step, data in enumerate(train_loader_1):
			x = data[0]
			y = data[1]
			length = data[2]

			# pack_padded x
			x = rnn_utils.pack_padded_sequence(x, length, batch_first=True)
			y = torch.from_numpy(y.numpy().reshape(-1,1))
			output = lstm(x)
			loss = loss_func(output, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if step % 10 == 0:
				lstm = lstm.eval()
				with torch.no_grad():
					x = data[0]
					y = data[1]
					length = data[2]
					x = rnn_utils.pack_padded_sequence(x, length, batch_first=True)
					y = torch.from_numpy(y.numpy().reshape(-1,1))
					output = lstm(x)
					loss = loss_func(output, y)
					print('step:',step,'|train_loss: %.5f ' % loss.data.numpy(),'\n')
					print('pred_y:',to_one_zero(torch.sigmoid(output).numpy()))
					print('\ntrue_y:',y.numpy().reshape(-1,))