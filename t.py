import pandas as pd
import numpy as np
import math
import torch
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
from torch import nn
from sklearn import preprocessing
from torch.autograd import Variable
import torch.nn.functional as F

X_h = []
X_nh = []

def load_h(x, y):
	while x <= y:
		try:
			path = './h/h' + str(x) + '.csv'
			data = pd.read_csv(path, header=None)
			# set TimeStep
			if len(data) > 50:
				data = data.iloc[:50, 3:15]
				# print(data)
				data = data.values.tolist()
				X_h.append(data)
		except:
			pass
		x += 1

def load_nh(x, y):
	while x <= y:
		try:
			path = './nh/nh' + str(x) + '.csv'
			data = pd.read_csv(path, header=None)
			# set TimeStep
			if len(data) > 50:
				data = data.iloc[:50, 3:15]
				# print(data)
				data = data.values.tolist()
				X_nh.append(data)
		except:
			pass
		x += 1


def to_one_zero(X):
	for i in range(len(X)):
		if X[i][0] >= 0.5:
			X[i][0] = 1
		else:
			X[i][0] = 0
	X = X.reshape(-1,)
	return X

BATCH_SIZE = 5
TIME_STEP = 50
INTPUT_SIZE = 12
LR = 0.001

class LSTM(nn.Module):
	def __init__(self):
		super(LSTM, self).__init__()
		self.linear_layers= nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(INTPUT_SIZE, 24),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(24, 8),
			nn.ReLU(),
			)
		self.rnn = nn.LSTM(
			input_size=8,
			hidden_size=16,
			num_layers=2,
			batch_first=True,
			)
		self.layer2 = nn.Linear(16,1)

	def forward(self, x):
		out = self.linear_layers(x)
		h0 = c0 = torch.randn(2, x[0]., 16)
		r_out, _ = self.rnn(out, (h0, c0))
		r_out = r_out[:,-1,:]
		out = self.layer2(r_out)
		# out = self.softmax(out)
		# out = torch.sigmoid(out)
		return out



if __name__ == '__main__':

	# 标签 = 1的样本
	load_h(2,237)
	# 标签 = 0的样本
	load_nh(2,520)
	X_h = np.array(X_h)
	X_nh = np.array(X_nh)
	X_h = torch.from_numpy(X_h)
	X_nh = torch.from_numpy(X_nh)
	print(X_h.shape) # (211, 50, 12)
	print(X_nh.shape) # (471, 50, 12)

	y_h = torch.ones(len(X_h))
	y_nh = torch.zeros(len(X_nh))

	X = torch.cat((X_h, X_nh), 0).type(torch.FloatTensor).numpy()
	y = torch.cat((y_h, y_nh),).type(torch.LongTensor).numpy()

	# normalize
	X = X.reshape(-1, 12)
	min_max_scaler = preprocessing.MinMaxScaler()
	X = min_max_scaler.fit_transform(X)
	X = X.reshape(-1, 50, 12)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
	X_train = torch.from_numpy(X_train).float()
	X_test = torch.from_numpy(X_test).float()
	y_train = torch.from_numpy(y_train).float()
	y_test = torch.from_numpy(y_test).float()

	torch_dataset = Data.TensorDataset(X_train, y_train)
	loader = Data.DataLoader(
		dataset=torch_dataset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=2,
		)

	lstm = LSTM()
	# print(lstm)
	optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
	# loss_func = nn.CrossEntropyLoss()
	loss_func = nn.BCEWithLogitsLoss()

	# training and testing
	# for epoch in range(3):
	# 	for step, (b_x, b_y) in enumerate(loader):
	# 		train data...
	# 		print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
	# 			b_x.numpy(), '\n batch y: ', b_y.numpy())
	# 		print('lenx: ', len(b_x), 'leny: ',len(b_y), '| batch y: ', b_y.numpy())
	# 		b_x = b_x.view(-1, 50, 12)

	# 		output = lstm(b_x)
	# 		loss = loss_func(output, b_y)
	# 		print('train loss: %.4f' % loss.data.numpy())
	# 		pred_y = torch.max(output, 1)[1].data.numpy()
	# 		print('pred_y:',pred_y,'y:',b_y.data.numpy(),'train loss: %.4f' % loss.data.numpy())


	# 		if step % 50 == 0:
	# 			lstm = lstm.eval()
	# 			with torch.no_grad():
	# 				test_output = lstm(X_test)
	# 				# 取pred_y行最大值的下标
	# 				pred_y = torch.max(test_output, 1)[1].data.numpy()
	# 				print('Epoch: ', epoch, '\n| pred_y: ', pred_y)
	# 				accuracy = float((pred_y == y_test).astype(int).sum()) / float(y_test.size)
	# 				print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

	for epoch in range(3):
		for step, (b_x, b_y) in enumerate(loader):
			b_x = b_x.view(-1, 50, 12)
			output = lstm(b_x)
			b_y = b_y.numpy().reshape(-1,1)
			b_y = torch.from_numpy(b_y)
			loss = loss_func(output, b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print(loss.data.numpy())

			if step % 50 == 0:
				lstm = lstm.eval()
				with torch.no_grad():
					test_output = lstm(X_test.view(-1,50,12))
					y_test = y_test.numpy().reshape(-1,1)
					y_test = torch.from_numpy(y_test)
					test_loss = loss_func(test_output, y_test)
					true_y = y_test.numpy().reshape(-1,)
					pred_y = to_one_zero(torch.sigmoid(test_output).numpy())
					print('epoch',epoch + 1,'|step',step,'|test_loss: %.5f ' % test_loss.data.numpy(),
						'|AUC: %.5f ' % roc_auc_score(y_test,torch.sigmoid(test_output)), '|accu: %.5f ' % accuracy_score(pred_y, true_y))
