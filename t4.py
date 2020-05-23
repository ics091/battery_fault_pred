import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import random
from matplotlib import pyplot as plt

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
			if len(data) > 30:
				data = data.iloc[:30, 1:15].values.tolist()
				cb_data.append(data)
				label.append(1)
		except:
			pass
		x += 1

	# print(len(label))

	while y <= l2:
		try:
			p = p2 + str(y) + '.csv'
			data = pd.read_csv(p, header=None)
			if len(data) > 30:
				data = data.iloc[:30, 1:15].values.tolist()
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

	data_1908, label_1908 = load_HNH(55,98,time_list[3])
	data_1909, label_1909 = load_HNH(25,62,time_list[4])
	data_1910, label_1910 = load_HNH(45,90,time_list[5])

	X1 = []
	y1 = []
	X2 = []
	y2 = []

	for x, y in zip(data_1811, label_1811):
		X1.append(x)
		y1.append(y)

	for x, y in zip(data_1812, label_1812):
		X1.append(x)
		y1.append(y)

	for x, y in zip(data_1901, label_1901):
		X1.append(x)
		y1.append(y)

	for x, y in zip(data_1908, label_1908):
		X2.append(x)
		y2.append(y)

	for x, y in zip(data_1909, label_1909):
		X2.append(x)
		y2.append(y)

	for x, y in zip(data_1910, label_1910):
		X2.append(x)
		y2.append(y)

	return X1, y1, X2, y2

def pre_data(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	X_train = X_train.reshape(-1,14)
	X_test = X_test.reshape(-1,14)
	mean_std_scaler = preprocessing.StandardScaler().fit(X_train)
	X_train = mean_std_scaler.transform(X_train).reshape(-1,30,14)
	X_test = mean_std_scaler.transform(X_test).reshape(-1,30,14)
	X_train = torch.from_numpy(X_train).float()
	X_test = torch.from_numpy(X_test).float()
	y_train = torch.from_numpy(y_train).float()
	y_test = torch.from_numpy(y_test).float()

	return X_train, X_test, y_train, y_test


# pred result to one zero
def to_one_zero(X):
	for i in range(len(X)):
		if X[i][0] >= 0.5:
			X[i][0] = 1
		else:
			X[i][0] = 0
	X = X.reshape(-1,)
	return X

class LSTM(nn.Module):
	def __init__(self):
		super(LSTM, self).__init__()
		# self.linear_layers= nn.Sequential(
		# 	nn.Dropout(0.5),
		# 	nn.Linear(INTPUT_SIZE, 8),
		# 	nn.ReLU(),
		# 	)
		# self.rnn = nn.LSTM(
		# 	input_size=8,
		# 	dropout=0.5,
		# 	hidden_size=16,
		# 	num_layers=2,
		# 	batch_first=True,
		# 	)
		# self.dense= nn.Sequential(
		# 	nn.Dropout(0.5),
		# 	nn.Linear(16, 6),
		# 	nn.ReLU(),
		# 	nn.Linear(6,1)
		# 	)

		self.cnn = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(INTPUT_SIZE,8,2),
			nn.ReLU(),
			)
		self.rnn = nn.LSTM(
			input_size=8,
			dropout=0.5,
			hidden_size=16,
			num_layers=2,
			batch_first=True,
			)
		self.dense= nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(16, 6),
			nn.ReLU(),
			nn.Linear(6,1)
			)

	def forward(self, x):
		# out = self.linear_layers(x)
		# h0 = c0 = torch.randn(2, x.shape[0], 16)
		# r_out, _ = self.rnn(out, (h0, c0))
		# r_out = r_out[:,-1,:]
		# out = self.dense(r_out)
		# return out
		x = x.permute(0,2,1)
		out = self.cnn(x)
		out = out.permute(0,2,1)
		h0 = c0 = torch.randn(2, x.shape[0], 16)
		r_out, _ = self.rnn(out, (h0, c0))
		r_out = r_out[:,-1,:]
		out = self.dense(r_out)
		return out


BATCH_SIZE = 32
TIME_STEP = 30
INTPUT_SIZE = 14
LR = 0.001

if __name__ == '__main__':
	X1, y1, X2, y2 = load_data()
	X1 = np.array(X1) # (299, 30, 12)
	y1 = np.array(y1) # (299,)
	X2 = np.array(X2) # (375, 30, 12)
	y2 = np.array(y2) # (375,)
	print(X1.shape)
	print(X2.shape)

	X1_train, X1_test, y1_train, y1_test = pre_data(X1,y1)
	X2_train, X2_test, y2_train, y2_test = pre_data(X2,y2)

	dataset_1 = Data.TensorDataset(X1_train, y1_train)
	dataset_2 = Data.TensorDataset(X2_train, y2_train)
	loader = Data.DataLoader(
		dataset=dataset_1,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=2,
		)

	lstm = LSTM()
	optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
	loss_func = nn.BCEWithLogitsLoss()
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)

	epoch_rec = []
	train_loss_rec = []
	loss_rec = []
	AUC_rec = []
	accu_rec = []

	for epoch in range(200):
		for step, (b_x, b_y) in enumerate(loader):

			b_x = b_x.view(-1, 30, 14)
			output = lstm(b_x)
			b_y = b_y.numpy().reshape(-1,1)
			b_y = torch.from_numpy(b_y)
			loss = loss_func(output, b_y)
			train_loss = loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# print(loss.data.numpy())

			if step % 50 == 0:
				lstm = lstm.eval()
				with torch.no_grad():
					test_output = lstm(X1_test.view(-1,30,14))
					y_test = y1_test.numpy().reshape(-1,1)
					y_test = torch.from_numpy(y_test)
					test_loss = loss_func(test_output, y_test)
					true_y = y_test.numpy().reshape(-1,)
					pred_y = to_one_zero(torch.sigmoid(test_output).numpy())
					print('epoch',epoch + 1,'|step',step,'|loss: %.5f ' % test_loss.data.numpy(),
						'|AUC: %.5f ' % roc_auc_score(y_test,torch.sigmoid(test_output)),
						'|accu: %.5f ' % accuracy_score(pred_y, true_y))
					# print('pred_y:',pred_y)
					# print('true_y:',true_y)

					epoch_rec.append(epoch)
					train_loss_rec.append(train_loss.data.numpy())
					loss_rec.append(test_loss.data.numpy())
					AUC_rec.append(roc_auc_score(y_test,torch.sigmoid(test_output)))
					accu_rec.append(accuracy_score(pred_y, true_y))
				lstm.train()

	epoch_rec = np.array(epoch_rec)
	loss_rec = np.array(loss_rec)
	train_loss_rec = np.array(train_loss_rec)
	AUC_rec = np.array(AUC_rec)
	accu_rec = np.array(accu_rec)
	#plt.plot(epoch_rec,train_loss_rec,label='train-loss')
	plt.plot(epoch_rec,loss_rec,label='log-loss')
	plt.plot(epoch_rec,AUC_rec,label='AUC')
	# plt.plot(epoch_rec,accu_rec,label='accu')
	plt.xlabel('epoch')
	plt.ylabel('logloss & AUC')
	plt.legend()
	plt.show()