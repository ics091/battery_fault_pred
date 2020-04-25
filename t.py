import pandas as pd
import numpy as np
import math
import torch
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
from torch import nn

X_h = []
X_nh = []

def load_h(x, y):
	while x <= y:
		try:
			path = './h/h' + str(x) + '.csv'
			data = pd.read_csv(path, header=None)
			# set TimeStep
			if len(data) > 30:
				data = data.iloc[:30, 3:17]
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
			if len(data) > 30:
				data = data.iloc[:30, 3:17]
				# print(data)
				data = data.values.tolist()
				X_nh.append(data)
		except:
			pass
		x += 1

BATCH_SIZE = 5
TIME_STEP = 30
INTPUT_SIZE = 14
LR = 0.01

class LSTM(nn.Module):
	def __init__(self):
		super(LSTM, self).__init__()

		self.rnn = nn.LSTM(
			input_size = INTPUT_SIZE,
			hidden_size = 28,
			num_layers = 1,
			batch_first=True,
			)

		self.out = nn.Linear(28, 2)

	def forward(self, x):
		r_out, (h_n, h_c) = self.rnn(x, None)
		# 选取最后一个时间点的 r_out 输出
		# 这里 r_out[:, -1, :] 的值也是 h_n 的值
		out = self.out(r_out[:, -1, :])
		return out

if __name__ == '__main__':
	load_h(2,237)
	load_nh(2,520)
	X_h = np.array(X_h)
	X_nh = np.array(X_nh)
	X_h = torch.from_numpy(X_h)
	X_nh = torch.from_numpy(X_nh)
	# print(X_h.shape) # (219, 30, 14)
	# print(X_nh.shape) # (487, 30, 14)

	y_h = torch.ones(len(X_h))
	y_nh = torch.zeros(len(X_nh))

	X = torch.cat((X_h, X_nh), 0).type(torch.FloatTensor).numpy()
	y = torch.cat((y_h, y_nh),).type(torch.LongTensor).numpy()

	# print(X.shape)
	# print(y.shape)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	X_train = torch.from_numpy(X_train)
	X_test = torch.from_numpy(X_test)
	y_train = torch.from_numpy(y_train)
	# y_test = torch.from_numpy(y_test)

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
	loss_func = nn.CrossEntropyLoss()

	# training and testing
	for epoch in range(10):
		for step, (b_x, b_y) in enumerate(loader):
			# train data...
			# print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
			# 	b_x.numpy(), '| batch y: ', b_y.numpy())
			b_x = b_x.view(-1, 30, 14)

			output = lstm(b_x)
			loss = loss_func(output, b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if step % 50 == 0:
				test_output = lstm(X_test)
				pred_y = torch.max(test_output, 1)[1].data.numpy()
				print('Epoch: ', epoch, '| pred_y: ', pred_y)
				# accuracy = float((pred_y == y_test).astype(int).sum()) / float(y_test.size)
				# print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)